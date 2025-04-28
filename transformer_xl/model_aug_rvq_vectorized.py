import tensorflow as tf
import numpy as np
import modules_rvq
import utils
import time
import os
import random
from copy import deepcopy
from sklearn.cluster import KMeans
import nvidia_smi

nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
vram_used = info.used // (1024 * 1024)  # in MB

#tf.config.optimizer.set_jit(True)  # Enable XLA globally


class TransformerXL(object):
    ########################################
    # initialize
    ########################################
    def __init__(self, event2word, word2event, checkpoint=None, is_training=False, training_seqs=None, rvq_quantizers=4, codebook_size=256):
        # load dictionary
        self.event2word = event2word
        self.word2event = word2event
        # model settings
        self.x_len = 512      #input sequence length
        self.mem_len = 512
        self.n_layer = 12
        self.d_embed = 512
        self.d_model = 512
        self.dropout = 0.1
        self.n_head = 8
        self.d_head = self.d_model // self.n_head
        self.d_ff = 2048
        self.n_token = len(self.event2word)
        self.learning_rate = 2e-4
        self.group_size = 3
        self.entry_len = self.group_size * self.x_len
        # mode
        self.is_training = is_training
        self.training_seqs = training_seqs
        self.checkpoint = checkpoint
        if self.is_training: # train from scratch or finetune
            self.batch_size = 8
        else: # inference
            self.batch_size = 1

        # Add RVQ parameters
        self.rvq_quantizers = rvq_quantizers
        self.codebook_size = codebook_size
        
        # load model
        self.load_model()

        # Performance monitoring
        self.epoch_times = []
        self.vram_log = []
    '''    
    def vectorized_rvq(self, emb):
        """Vectorized RVQ implementation using tensor ops"""
        # Stack codebooks [num_quantizers, codebook_size, dim]
        cb_stack = tf.stack(self.codebooks)
        
        # Initialize residuals and quantized output
        residuals = tf.expand_dims(emb, 1)  # [seq, 1, batch, dim]
        quantized = tf.zeros_like(emb)
        
        # Vectorized residual processing
        for i in range(self.rvq_quantizers):
            # Compute distances [seq, batch, codebook_size]
            dists = tf.norm(
                residuals - cb_stack[i], 
                axis=-1
            )
            
            # Find indices [seq, batch]
            indices = tf.argmin(dists, axis=-1)
            
            # Gather quantized vectors [seq, batch, dim]
            q_vectors = tf.gather(cb_stack[i], indices)
            
            # Update residuals and quantized output
            residuals -= q_vectors
            quantized += q_vectors

        return quantized
    '''

    def vectorized_rvq(self, emb):
        # Stack codebooks: [num_quantizers, codebook_size, d_model]
        cb_stack = tf.stack(self.codebooks)  # [4, 256, 512]
        
        # emb: [seq_len, batch_size, d_model]
        seq_len = tf.shape(emb)[0]
        batch_size = tf.shape(emb)[1]
        
        # Prepare for quantization
        residual = emb  # [seq_len, batch_size, d_model]
        quantized_list = []
        
        for i in range(self.rvq_quantizers):
            # cb: [codebook_size, d_model]
            cb = cb_stack[i]  # [256, 512]
            # Expand dims for broadcasting
            res_exp = tf.expand_dims(residual, 2)  # [seq_len, batch_size, 1, d_model]
            cb_exp = tf.reshape(cb, [1, 1, self.codebook_size, self.d_model])  # [1, 1, 256, 512]
            # Compute L2 distance
            dists = tf.norm(res_exp - cb_exp, axis=-1)  # [seq_len, batch_size, 256]
            # Find nearest codebook entry
            indices = tf.argmin(dists, axis=-1)  # [seq_len, batch_size]
            # Gather quantized vectors
            q_vec = tf.gather(cb, indices)  # [seq_len, batch_size, d_model]
            quantized_list.append(q_vec)
            # Update residual
            residual = residual - q_vec

        # Sum all quantized vectors (residual quantization)
        quantized = tf.add_n(quantized_list)  # [seq_len, batch_size, d_model]
        return quantized


    #@tf.function(experimental_compile=True)

    def residual_quantize(self, emb):
        return self.vectorized_rvq(emb)


    ########################################
    # load model
    ########################################
    
    def load_model(self):
        tf.compat.v1.disable_eager_execution()
        
        # Placeholders
        self.x = tf.compat.v1.placeholder(tf.int32, [self.batch_size, None])
        self.y = tf.compat.v1.placeholder(tf.int32, [self.batch_size, None])
        self.mems_i = [tf.compat.v1.placeholder(tf.float32, [self.mem_len, self.batch_size, self.d_model]) 
                      for _ in range(self.n_layer)]

        # Model components
        self.global_step = tf.compat.v1.train.get_or_create_global_step()
        initializer = tf.compat.v1.initializers.random_normal(stddev=0.02)
        proj_initializer = tf.compat.v1.initializers.random_normal(stddev=0.01)

        with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope()):
            # Embedding and codebooks
            self.embedding = tf.compat.v1.get_variable(
                "embedding", 
                [self.n_token, self.d_model],
                initializer=initializer
            )
            
            # Initialize codebooks with k-means
            self.codebooks = [
                tf.compat.v1.get_variable(
                    f"codebook_{i}",
                    [self.codebook_size, self.d_model],
                    initializer=tf.compat.v1.initializers.zeros(),
                    trainable=True
                ) for i in range(self.rvq_quantizers)
            ]

            # Input processing
            xx = tf.transpose(self.x, [1, 0])
            yy = tf.transpose(self.y, [1, 0])
            
            # RVQ pipeline
            emb = tf.nn.embedding_lookup(self.embedding, xx)
            quant_emb = self.residual_quantize(emb)

            # Modified transformer call
            loss, self.logits, self.new_mem = modules_rvq.transformer(
                dec_inp=quant_emb,
                target=yy,
                mems=self.mems_i,
                n_token=self.n_token,
                n_layer=self.n_layer,
                d_model=self.d_model,
                d_embed=self.d_embed,
                n_head=self.n_head,
                d_head=self.d_head,
                d_inner=self.d_ff,
                dropout=self.dropout,
                dropatt=self.dropout,
                initializer=initializer,
                proj_initializer=proj_initializer,
                is_training=self.is_training,
                mem_len=self.mem_len,
                skip_embed=True
            )

            # Loss components
            self.commit_loss = tf.reduce_mean(tf.square(quant_emb - emb))
            self.avg_loss = tf.reduce_mean(loss) + 0.25 * self.commit_loss

        # Optimizer with gradient clipping
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
        grads = tf.gradients(self.avg_loss, tf.compat.v1.trainable_variables())
        grads, _ = tf.clip_by_global_norm(grads, 100)
        self.train_op = self.optimizer.apply_gradients(
            zip(grads, tf.compat.v1.trainable_variables()),
            global_step=self.global_step
        )

        # Checkpoint handling
        self.saver = tf.compat.v1.train.Saver(
            var_list=[v for v in tf.compat.v1.global_variables() 
                     if 'codebook' not in v.name],
            max_to_keep=100
        )
        
        config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)
        )
        self.sess = tf.compat.v1.Session(config=config)
        
        if self.checkpoint:
            self.saver.restore(self.sess, self.checkpoint)
            # Initialize codebooks with k-means if not present
            self.initialize_codebooks()
        else:
            self.sess.run(tf.compat.v1.global_variables_initializer())
            self.initialize_codebooks()

    def initialize_codebooks(self):
        """Proper codebook initialization with sufficient samples"""
        from sklearn.cluster import MiniBatchKMeans
        
        # Collect sufficient samples (4x codebook_size)
        num_samples = max(1024, self.codebook_size * 4)
        sample_indices = []
        
        # Aggregate samples from multiple sequences
        for seq in self.training_seqs[:num_samples//self.x_len + 1]:
            sample_indices.extend(seq[:self.x_len])
            if len(sample_indices) >= num_samples:
                break
        
        # Get embeddings [num_samples, 512]
        emb = self.sess.run(
            tf.nn.embedding_lookup(self.embedding, sample_indices[:num_samples])
        )
        
        # Flatten to [num_samples, d_model]
        flat_emb = emb.reshape(-1, self.d_model)
        
        # Handle case where we still don't have enough samples
        if flat_emb.shape[0] < self.codebook_size:
            repeat_factor = (self.codebook_size // flat_emb.shape[0]) + 1
            flat_emb = np.tile(flat_emb, (repeat_factor, 1))[:self.codebook_size*2]

        # Initialize codebooks sequentially with residuals
        for i, cb in enumerate(self.codebooks):
            # MiniBatchKMeans is more efficient for large codebooks
            kmeans = MiniBatchKMeans(n_clusters=self.codebook_size, 
                                    batch_size=512,
                                    n_init=3)
            
            kmeans.fit(flat_emb)
            
            # Assign centroids to codebook
            centroids = kmeans.cluster_centers_.astype(np.float32)
            self.sess.run(tf.compat.v1.assign(cb, centroids))
            
            # Update residuals for next codebook
            distances = np.linalg.norm(flat_emb[:, None] - centroids, axis=-1)
            nearest_indices = np.argmin(distances, axis=1)
            flat_emb -= centroids[nearest_indices]

    
    ########################################
    # data augmentation
    ########################################
    # return 
    def get_epoch_augmented_data(self, epoch, ep_start_pitchaug=10, pitchaug_range=(-3, 3)):
        pitchaug_range = [x for x in range(pitchaug_range[0], pitchaug_range[1] + 1)]
        training_data = []
        for seq in self.training_seqs:
            # pitch augmentation
            if epoch >= ep_start_pitchaug:
                seq = deepcopy(seq)
                pitch_change = random.choice( pitchaug_range )
                for i, ev in enumerate(seq):
                    #  event_id = 21 -> Note-On_21 : the lowest pitch on piano
                    if 'Note-On' in self.word2event[ev] and ev >= 21:
                        seq[i] += pitch_change
                    if 'Chord-Tone' in self.word2event[ev]:
                        seq[i] += pitch_change
                        # prevent pitch shift out of range
                        if seq[i] > self.event2word['Chord-Tone_B']:
                            seq[i] -= 12
                        elif seq[i] < self.event2word['Chord-Tone_C']:
                            seq[i] += 12
                    if 'Chord-Slash' in self.word2event[ev]:
                        seq[i] += pitch_change
                        # prevent pitch shift out of range
                        if seq[i] > self.event2word['Chord-Slash_B']:
                            seq[i] -= 12
                        elif seq[i] < self.event2word['Chord-Slash_C']:
                            seq[i] += 12

            # padding sequence to fit the entry length
            if len(seq) < self.entry_len + 1:
                padlen = self.entry_len - len(seq)
                seq.append(1)
                seq.extend([0 for x in range(padlen)])


            # first 10 epoch let the input include start or end of the song
            # -1 for assertion : len(seq) % self.entry_len == 1 (for x,y pair purpose)
            if epoch < 10:
              offset = random.choice([0, (len(seq) % self.entry_len) - 1]) # only 2 possible return value 
            else:
              offset = random.randint(0, (len(seq) % self.entry_len) - 1)  # all entries in the list are possible return value

            assert offset + 1 + self.entry_len * (len(seq) // self.entry_len) <= len(seq)

            seq = seq[ offset : offset + 1 + self.entry_len * (len(seq) // self.entry_len) ]

            assert len(seq) % self.entry_len == 1

            pairs = []
            for i in range(0, len(seq) - self.x_len, self.x_len):
                x, y = seq[i:i+self.x_len], seq[ i+1 : i+self.x_len+1 ]
                assert len(x) == self.x_len
                assert len(y) == self.x_len
                pairs.append([x, y])

            pairs = np.array(pairs)

            # put pairs into training data by groups
            for i in range(0, len(pairs) - self.group_size + 1, self.group_size):
                segment = pairs[i:i+self.group_size]
                assert len(segment) == self.group_size
                training_data.append(segment)

        training_data = np.array(training_data)

        # shuffle training data
        reorder_index = np.arange(len(training_data))
        np.random.shuffle( reorder_index )
        training_data = training_data[ reorder_index ]

        num_batches = len(training_data) // self.batch_size
        # training_data shape (666, 3, 2, 512)
        # training_data shape (group count, self.group_size, pair(x,y), 512)
        print ("training_data.shape , num_batches = {} , {}".format(training_data.shape,num_batches))
        return training_data, num_batches

    ########################################
    # train w/ augmentation
    ########################################
    def train_augment(self, output_checkpoint_folder, pitchaug_range=(-3, 3), logfile=None):

        assert self.training_seqs is not None

        # check output folder
        if not os.path.exists(output_checkpoint_folder):
            os.mkdir(output_checkpoint_folder)

        # check log file folder
        if logfile:
            if not os.path.dirname(logfile) == "":
                os.makedirs(os.path.dirname(logfile),exist_ok=True) 
        
        st = time.time()

        for e in range(0, 50):
            epoch_start = time.time()
            # one epoch
            # get all data with augmentation
            training_data, num_batches = self.get_epoch_augmented_data(e)
            
            total_loss = []
            for i in range(num_batches):
                # in one batch
                # get one batch data
                segments = training_data[self.batch_size*i:self.batch_size*(i+1)]

                # memory cache for all layers of tranformer
                batch_m = [np.zeros((self.mem_len, self.batch_size, self.d_model), dtype=np.float32) for _ in range(self.n_layer)]
                
                for j in range(self.group_size):
                    batch_x = segments[:, j, 0, :]
                    batch_y = segments[:, j, 1, :]
                    # prepare feed dict
                    # self.x = tf.compat.v1.placeholder(tf.int32, shape=[self.batch_size, None])
                    # self.y = tf.compat.v1.placeholder(tf.int32, shape=[self.batch_size, None])
                    feed_dict = {self.x: batch_x, self.y: batch_y}

                    # self.mems_i a placeholder for memory of all layers in transformer
                    # self.mems_i = [tf.compat.v1.placeholder(tf.float32, [self.mem_len, self.batch_size, self.d_model]) for _ in range(self.n_layer)]
                    
                    

                    for m, m_np in zip(self.mems_i, batch_m):
                        feed_dict[m] = m_np

                    # run
                    _, gs_, loss_, new_mem_ = self.sess.run([self.train_op, self.global_step, self.avg_loss, self.new_mem], feed_dict=feed_dict)
                    
                    batch_m = new_mem_
                    total_loss.append(loss_)
                    
                    # print ('Current lr: {}'.format(self.sess.run(self.optimizer._lr)))
                    print('>>> Epoch: {}, Step: {}, Loss: {:.5f}, Time: {:.2f}'.format(e, gs_, loss_, time.time()-st))

            current_emb = tf.nn.embedding_lookup(self.embedding, self.x)  # Get current batch embeddings
            used = [
                tf.math.count_nonzero(
                    tf.unique(
                        tf.argmin(
                            tf.norm(
                                tf.stop_gradient(current_emb) - cb,  # Add stop_gradient for stability
                                axis=-1
                            ),
                            axis=-1
                        )
                    )[0]
                ) for cb in self.codebooks
            ]
            print(f"Codebook usage: {[u/self.codebook_size for u in used]}")


            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)

            # VRAM logging using pynvml
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            vram_used = info.used // (1024 * 1024)  # in MB
            self.vram_log.append(vram_used)

            print(f'[epoch {e} avg loss] {np.mean(total_loss):.5f} | VRAM: {vram_used}MB')
            
            if e >= 0:
                
                if e % 10 == 0: #or 0.085 <= np.mean(total_loss) <= 0.10 or np.mean(total_loss) <= 0.8:
                    self.saver.save(self.sess, '{}/model-{:03d}-{:.3f}'.format(output_checkpoint_folder, e, np.mean(total_loss)))
                if logfile:
                    with open(logfile, 'a') as f:
                        f.write('epoch = {:03d} | loss = {:.5f} | time = {:.2f}\n'.format(e, np.mean(total_loss), time.time()-st))
            # stop
            if np.mean(total_loss) <= 0.05:
                break

    ########################################
    # train
    ########################################
    def train(self, training_data, output_checkpoint_folder):
        # check output folder
        if not os.path.exists(output_checkpoint_folder):
            os.mkdir(output_checkpoint_folder)
        # shuffle
        index = np.arange(len(training_data))
        np.random.shuffle(index)
        training_data = training_data[index]
        num_batches = len(training_data) // self.batch_size
        st = time.time()
        for e in range('noo'):
            total_loss = []
            for i in range(num_batches):
                segments = training_data[self.batch_size*i:self.batch_size*(i+1)]
                batch_m = [np.zeros((self.mem_len, self.batch_size, self.d_model), dtype=np.float32) for _ in range(self.n_layer)]
                for j in range(self.group_size):
                    batch_x = segments[:, j, 0, :]
                    batch_y = segments[:, j, 1, :]
                    # prepare feed dict
                    feed_dict = {self.x: batch_x, self.y: batch_y}
                    for m, m_np in zip(self.mems_i, batch_m):
                        feed_dict[m] = m_np
                    # run
                    _, gs_, loss_, new_mem_ = self.sess.run([self.train_op, self.global_step, self.avg_loss, self.new_mem], feed_dict=feed_dict)
                    batch_m = new_mem_
                    total_loss.append(loss_)
                    # print ('Current lr: {}'.format(self.sess.run(self.optimizer._lr)))
                    print('>>> Epoch: {}, Step: {}, Loss: {:.5f}, Time: {:.2f}'.format(e, gs_, loss_, time.time()-st))

            print ('[epoch {} avg loss] {:.5f}'.format(e, np.mean(total_loss)))
            if not e % 6:
                self.saver.save(self.sess, '{}/model-{:03d}-{:.3f}'.format(output_checkpoint_folder, e, np.mean(total_loss)))
            # stop
            if np.mean(total_loss) <= 0.05:
                break

    ########################################
    # search strategy: temperature (re-shape)
    ########################################
    def temperature(self, logits, temperature):
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        return probs

    ########################################
    # search strategy: topk (truncate)
    ########################################
    def topk(self, probs, k):
        sorted_index = np.argsort(probs)[::-1]
        candi_index = sorted_index[:k]
        candi_probs = [probs[i] for i in candi_index]
        candi_probs /= sum(candi_probs)
        word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
        return word

    ########################################
    # search strategy: nucleus (truncate)
    ########################################
    def nucleus(self, probs, p):
        probs /= sum(probs)
        sorted_probs = np.sort(probs)[::-1]
        sorted_index = np.argsort(probs)[::-1]
        cusum_sorted_probs = np.cumsum(sorted_probs)
        after_threshold = cusum_sorted_probs > p
        if sum(after_threshold) > 0:
            last_index = np.where(after_threshold)[0][-1]
            candi_index = sorted_index[:last_index]
        else:
            candi_index = sorted_index[:3] # just assign a value
        candi_probs = [probs[i] for i in candi_index]
        candi_probs /= sum(candi_probs)
        word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
        return word

    ########################################
    # inference (for batch size = 1)
    ########################################
    def inference(self, n_bars, strategies, params, use_structure=False, init_mem=None):
        print("Start model inference...")
        start_time = time.time()
        # initial start
        words = [[]]
        # add new part if needed
        if use_structure:
            words[-1].append( self.event2word['Part-Start_I'] if random.random() > 0.5 else self.event2word['Part-Start_A'] )
            words[-1].append( self.event2word['Rep-Start_1'])
        # add new bar
        words[-1].append( self.event2word['Bar'] )
        # add position 0
        words[-1].append( self.event2word['Position_0/64'] ) 
        # add random tempo class and bin
        chosen_tempo_cls = random.choice([x for x in range(0, 5)])
        words[-1].append( self.event2word['Tempo-Class_{}'.format(chosen_tempo_cls)] )
        tempo_bin_start = chosen_tempo_cls * 12
        words[-1].append( random.choice(
            [tb for tb in range(self.event2word['Tempo_50.00'] + tempo_bin_start, self.event2word['Tempo_50.00'] + tempo_bin_start + 12)]
        ))
        # add random chord
        if not use_structure and random.random() > 0.5 or \
           use_structure and words[-1][0] == self.event2word['Part-Start_A']:
            words[-1].append( random.choice(
                [ct for ct in range(self.event2word['Chord-Tone_C'], self.event2word['Chord-Tone_C'] + 12)]
            ))
            words[-1].append( random.choice(
                [self.event2word[ct] for ct in self.event2word.keys() if 'Chord-Type' in ct]
            ))
        # initialize mem
        if init_mem is None:
            batch_m = [np.zeros((self.mem_len, self.batch_size, self.d_model), dtype=np.float32) for _ in range(self.n_layer)]
        else:
            batch_m = init_mem
        
        # generate

        
        initial_flag, initial_cnt = True, 0
        generated_bars = 0

        # define legal beat posisition
        beat_pos = set(['Position_0/64', 'Position_16/64', 'Position_32/64', 'Position_48/64'])

        allowed_pos = set([x for x in range(self.event2word['Position_0/64'] + 1, self.event2word['Position_0/64'] + 17)])
        fail_cnt = 0
        
        while generated_bars < n_bars:
            print("Generating bars #{}/{}".format(generated_bars+1,n_bars), end='\r')
            if fail_cnt:
                print ('failed iterations:', fail_cnt)
            
            if fail_cnt > 256:
                print ('model stuck ...')
                exit()

            # prepare input
            if initial_flag:
                temp_x = np.zeros((self.batch_size, len(words[0])))
                for b in range(self.batch_size):
                    for z, t in enumerate(words[b]):
                        temp_x[b][z] = t
                initial_flag = False
            else:
                temp_x = np.zeros((self.batch_size, 1))
                for b in range(self.batch_size):
                    temp_x[b][0] = words[b][-1]

            # prepare feed dict
            # inside a feed dict
            # placeholder : data
            # put input into feed_dict
            feed_dict = {self.x: temp_x}

            # put memeory into feed_dict
            for m, m_np in zip(self.mems_i, batch_m):
                feed_dict[m] = m_np
            
            # model (prediction)
            _logits, _new_mem = self.sess.run([self.logits, self.new_mem], feed_dict=feed_dict)

            logits = _logits[-1, 0]

            # temperature or not
            if 'temperature' in strategies:
                if initial_flag:
                    probs = self.temperature(logits=logits, temperature=1.5)
                else:
                    probs = self.temperature(logits=logits, temperature=params['t'])
            else:
                probs = self.temperature(logits=logits, temperature=1.)

            # sampling
            # word : the generated remi event
            word = self.nucleus(probs=probs, p=params['p'])
            # print("Generated new remi word {}".format(word))
            # skip padding
            if word in [0, 1]:
                fail_cnt += 1
                continue
            
            # illegal sequences 
            # words[0][-1] : last generated word
            if 'Bar' in self.word2event[words[0][-1]] and self.word2event[word] != 'Position_0/64':
                fail_cnt += 1
                continue
            if self.word2event[words[0][-1]] in beat_pos and 'Tempo-Class' not in self.word2event[word]:
                fail_cnt += 1
                continue
            if 'Tempo-Class' in self.word2event[words[0][-1]] and 'Tempo_' not in self.word2event[word]:
                fail_cnt += 1
                continue
            if 'Note-Velocity' in self.word2event[words[0][-1]] and 'Note-On' not in self.word2event[word]:
                fail_cnt += 1
                continue
            if 'Note-On' in self.word2event[words[0][-1]] and 'Note-Duration' not in self.word2event[word]:
                fail_cnt += 1
                continue
            if 'Chord-Tone' in self.word2event[words[0][-1]] and 'Chord-Type' not in self.word2event[word]:
                fail_cnt += 1
                continue
            if 'Chord-Type' in self.word2event[words[0][-1]] and 'Chord-Slash' not in self.word2event[word]:
                fail_cnt += 1
                continue
            if 'Position' in self.word2event[word] and word not in allowed_pos:
                fail_cnt += 1
                continue
            print("TEST!!!", word, type(word))
            if self.word2event[word].split('_')[0] == self.word2event[words[0][-1]].split('_')[0]:
                fail_cnt += 1
                continue

            
            # update allowed positions
            # if the new word is a beat event then we need to update the new allowed_pos
            # ex if the new word is (209: Position_16/64) then the allow_pos should update as following
            # ['Position_1/64' to  'Position_16/64']  -> ['Position_17/64' to  'Position_32/64']
            # exception: exceed the 64/64 go back to 0
            if self.word2event[word] in beat_pos:
                if self.word2event[word] == 'Position_48/64':
                    allowed_pos = set([x for x in range(self.event2word['Position_49/64'], self.event2word['Position_49/64'] + 15)] + [self.event2word['Position_0/64']])
                else:
                    allowed_pos = set([x for x in range(word + 1, word + 17)])

            # add new event to record sequence
            words[0].append(word)
            fail_cnt = 0

            # record n_bars
            if word == self.event2word['Bar']:
                generated_bars += 1
            # re-new mem
            batch_m = _new_mem

        print ('generated {} events'.format(len(words[0])))
        print(f'\nGenerated {n_bars} bars in {time.time()-start_time:.2f}s')
        return words[0]

    ########################################
    # close
    ########################################
    def close(self):
        self.sess.close()
