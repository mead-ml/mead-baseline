import tensorflow as tf

class PreprocessorCreator(object):
    def __init__(self, indices, lchars, upchars_lut, task, token_key, extra_feats):
        """
        indices are created during vocab creation.
        """
        self.word2index = indices['word']
        self.char2index = indices.get('char')
        self.indices = indices

        self.lchars = lchars
        self.upchars_lut = upchars_lut

        self.task = task
        self.token_key = token_key
        self.extra_feats = extra_feats

    def preproc_post(self, post_mappings):
        # Split the input string, assuming that whitespace is splitter
        # The client should perform any required tokenization for us and join on ' '

        # WARNING: This can be a bug if the user defaults the values (-1)
        # for conll, the mxlen=124, for idr, the mxlen is forced to a max BPTT
        # for twpos, the mxlen=38
        # this should probably be fixed by serializing the mxlen of the model
        # or rereading it from the tensor from file
        raw_post = post_mappings[self.token_key]
        # raw_post = post_mappings
        mxlen = self.task.config_params['preproc']['mxlen']
        mxwlen = self.task.config_params['preproc'].get('mxwlen')

        nraw_post = self._reform_raw(raw_post, mxlen)

        preprocs = {}
        words, sentence_length = self._create_word_vectors_from_post(nraw_post, mxlen)
        preprocs['word'] = words
        if 'char' in self.indices:
            chars, _ = self._create_char_vectors_from_post(nraw_post, mxlen, mxwlen)
            preprocs['char'] = chars


        for extra in self.extra_feats:
            index = self.indices[extra]
            nraw = self._reform_raw(post_mappings[extra], mxlen)
            t, _ = self._create_vectors_from_post(nraw, mxlen, index)
            preprocs[extra] = t

        return preprocs, sentence_length

    def _reform_raw(self, raw, mxlen):
        """
        Splits and rejoins a string to ensure that tokens meet
        the required max len.
        """
        #raw_post = tf.Print(raw_post, [raw_post])
        raw_tokens = tf.string_split(tf.reshape(raw, [-1])).values
        # sentence length <= mxlen
        nraw_post = tf.reduce_join(raw_tokens[:mxlen], separator=" ")

        return nraw_post

    def _create_word_vectors_from_post(self, nraw_post, mxlen):
        # vocab has only lowercase words
        split_chars = tf.string_split(tf.reshape(nraw_post, [-1]), delimiter="").values
        upchar_inds = self.upchars_lut.lookup(split_chars)
        lc_raw_post = tf.reduce_join(tf.map_fn(lambda x: tf.cond(x[0] > 25,
                                                                    lambda: x[1],
                                                                    lambda: self.lchars[x[0]]),
                                                (upchar_inds, split_chars), dtype=tf.string))
        word_tokens = tf.string_split(tf.reshape(lc_raw_post, [-1]))
        
        word_indices = self.word2index.lookup(word_tokens)
        
        # Reshape them out to the proper length
        reshaped_words = tf.sparse_reshape(word_indices, shape=[-1])
        sentence_length = tf.size(reshaped_words)  # tf.shape if 2 dims needed

        x = self._reshape_indices(reshaped_words, [mxlen])

        return x, sentence_length
    
    def _create_char_vectors_from_post(self, nraw_post, mxlen, mxwlen):
        # numchars per word should be <= mxwlen
        unchanged_word_tokens = tf.string_split(tf.reshape(nraw_post, [-1]))
        culled_word_token_vals = tf.substr(unchanged_word_tokens.values, 0, mxwlen)
        char_tokens = tf.string_split(culled_word_token_vals, delimiter='')

        char_indices = self.char2index.lookup(char_tokens)

        xch = self._reshape_indices(char_indices, [mxlen, mxwlen])

        sentence_length = tf.size(xch)

        return xch, sentence_length

    def _create_vectors_from_post(self, nraw_post, mxlen, index):
        tokens = tf.string_split(tf.reshape(nraw_post, [-1]))
        
        indices = index.lookup(tokens)
        
        # Reshape them out to the proper length
        reshaped = tf.sparse_reshape(indices, shape=[-1])
        sentence_length = tf.size(reshaped)  # tf.shape if 2 dims needed
        print(sentence_length)
        return self._reshape_indices(reshaped, [mxlen]), sentence_length

    def _reshape_indices(self, indices, shape):
        reshaped = tf.sparse_reset_shape(indices, new_shape=shape)

        # Now convert to a dense representation
        x = tf.sparse_tensor_to_dense(reshaped)
        x = tf.contrib.framework.with_shape(shape, x)

        return x
