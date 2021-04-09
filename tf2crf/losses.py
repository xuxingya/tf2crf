import tensorflow as tf
import tensorflow.keras.backend as K

class AdaptiveDscLoss(tf.keras.losses.Loss):
    """
    Dice coefficient for short, is an F1-oriented statistic used to gauge the similarity of two sets.
    Given two sets A and B, the vanilla dice coefficient between them is given as follows:
        Dice(A, B)  = 2 * True_Positive / (2 * True_Positive + False_Positive + False_Negative)
                    = 2 * |A and B| / (|A| + |B|)
    Math Function:
        U-NET: https://arxiv.org/abs/1505.04597.pdf
        dice_loss(p, y) = 1 - numerator / denominator
            numerator = 2 * \sum_{1}^{t} p_i * y_i + smooth
            denominator = \sum_{1}^{t} p_i + \sum_{1} ^{t} y_i + smooth
        if square_denominator is True, the denominator is \sum_{1}^{t} (p_i ** 2) + \sum_{1} ^{t} (y_i ** 2) + smooth
        V-NET: https://arxiv.org/abs/1606.04797.pdf
    Args:
        smooth (float, optional): a manual smooth value for numerator and denominator.
        square_denominator (bool, optional): [True, False], specifies whether to square the denominator in the loss function.
        with_logits (bool, optional): [True, False], specifies whether the input tensor is normalized by Sigmoid/Softmax funcs.
        ohem_ratio: max ratio of positive/negative, defautls to 0.0, which means no ohem.
        alpha: dsc alpha
    Shape:
        - input: (*)
        - target: (*)
        - mask: (*) 0,1 mask for the input sequence.
        - Output: Scalar loss
    Examples:
        loss = DiceLoss(with_logits=True, ohem_ratio=0.1)
        input = torch.FloatTensor([2, 1, 2, 2, 1])
        input.requires_grad=True
        target = torch.LongTensor([0, 1, 0, 0, 0])
        output = loss(input, target)
        output.backward()
    """
    def __init__(self, smooth=1, square_denominator=False, with_logits=True,ohem_ratio=0.3, alpha=0.01,index_label_position=True, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE):
        super(AdaptiveDscLoss, self).__init__()
        self.smooth: float = smooth
        self.square_denominator: bool = square_denominator
        self.with_logits: bool = with_logits
        self.ohem_ratio: float = ohem_ratio
        self.alpha: float = alpha
        self.index_label_position = index_label_position
        self.reduction = reduction

    def call(self, y_true, y_pred):
        logits_size = y_pred.shape[-1]
        y_true = tf.reshape(y_true, [-1])
        mask = tf.cast(tf.not_equal(y_true, 0), y_pred.dtype)
        if logits_size != 1:
            y_pred = tf.reshape(y_pred, [-1, logits_size])
            return self._multiple_class(y_pred, y_true, logits_size, mask)
        else:
            y_pred = tf.reshape(y_pred, [-1])
            return self._binary_class(y_pred, y_true, mask)

    @tf.function
    def _compute_dice_loss(self, flat_input, flat_target):
        flat_input = (flat_input+K.epsilon()) ** self.alpha * flat_input
        interection = K.sum(flat_input * flat_target, axis=-1)
        if not self.square_denominator:
            loss = 1 - tf.math.divide_no_nan((2 * interection + self.smooth), (K.sum(flat_input) + K.sum(flat_target) + self.smooth))
        else:
            loss = 1 - tf.math.divide_no_nan((2 * interection + self.smooth), (
                        K.sum(K.square(flat_input)) + K.sum(K.square(flat_target)) + self.smooth))
        return loss

    @tf.function
    def _multiple_class(self, input, target, logits_size, mask=None):
        flat_input = input
        flat_target = K.cast(K.one_hot(target,
                                       num_classes=logits_size), input.dtype) if self.index_label_position else K.cast(
            target, input.dtype)
        # values should be positive in flat_input
        flat_input = tf.math.softmax(flat_input, axis=-1) if self.with_logits else flat_input
        if mask is not None:
            flat_input = flat_input * K.expand_dims(mask, -1)
            flat_target = flat_target * K.expand_dims(mask, -1)
        else:
            mask = K.ones_like(target, dtype=input.dtype)
        loss = None
        if self.ohem_ratio > 0:
            mask_neg = tf.logical_not(K.cast(mask, 'bool'))
            for label_idx in range(logits_size):
                pos_example = K.cast(K.equal(target, label_idx), 'bool')
                neg_example = K.cast(K.not_equal(target, label_idx), 'bool')
                pos_num = K.sum(tf.cast(pos_example, 'int32'))
                neg_num = K.sum(K.cast(mask, 'int32')) - (
                            pos_num - K.sum(K.cast(tf.logical_and(mask_neg, pos_example), 'int32')))
                keep_num = tf.minimum(K.cast(K.cast(pos_num, input.dtype) * self.ohem_ratio / logits_size, 'int32'),
                                      neg_num)
                if keep_num > 0:
                    neg_scores = tf.boolean_mask(flat_input, neg_example)
                    neg_scores_idx = neg_scores[:, label_idx]
                    neg_scores_sort = tf.sort(neg_scores_idx)
                    threshold = neg_scores_sort[-keep_num + 1]
                    cond = ((tf.argmax(flat_input, axis=1) == label_idx) & (flat_input[:, label_idx] >= threshold)) | pos_example
                    ohem_mask_idx = tf.cast(cond, input.dtype)
                    flat_input_idx = flat_input[:, label_idx]
                    flat_target_idx = flat_target[:, label_idx]
                    flat_input_idx = flat_input_idx * ohem_mask_idx
                    flat_target_idx = flat_target_idx * ohem_mask_idx
                else:
                    flat_input_idx = flat_input[:, label_idx]
                    flat_target_idx = flat_target[:, label_idx]
                loss_idx = self._compute_dice_loss(flat_input_idx, flat_target_idx)
                if loss is None:
                    loss = loss_idx
                else:
                    loss += loss_idx
            return loss

        else:
            for label_idx in range(logits_size):
                # pos_example = tf.equal(target, label_idx)
                flat_input_idx = flat_input[:, label_idx]
                flat_target_idx = flat_target[:, label_idx]
                loss_idx = self._compute_dice_loss(flat_input_idx, flat_target_idx)
                if loss is None:
                    loss = loss_idx
                else:
                    loss += loss_idx
            return loss

    @tf.function
    def _binary_class(self, input, target, mask=None):
        flat_target = tf.cast(input, input.dtype)
        flat_input = tf.sigmoid(input) if self.with_logits else input

        if mask is not None:
            mask = tf.cast(mask, input)
            flat_input = flat_input * mask
            flat_target = flat_target * mask
        else:
            mask = tf.ones_like(target)
        if self.ohem_ratio > 0:
            pos_example = target > 0.5
            neg_example = target <= 0.5
            mask_neg_num = mask <= 0.5
            pos_num = K.sum(pos_example) - K.sum(pos_example & mask_neg_num)
            neg_num = K.sum(neg_example)
            keep_num = tf.minimum(tf.cast(pos_num * self.ohem_ratio), neg_num)
            neg_scores = tf.boolean_mask(flat_input, neg_example)
            neg_scores_sort = tf.sort(neg_scores)
            threshold = neg_scores_sort[-keep_num+1]
            cond = (flat_input > threshold) | pos_example
            ohem_mask = tf.cast(cond, 'int32')
            flat_input = flat_input * ohem_mask
            flat_target = flat_target * ohem_mask

        return self._compute_dice_loss(flat_input, flat_target)
