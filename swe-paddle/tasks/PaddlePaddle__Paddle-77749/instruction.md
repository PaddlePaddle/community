# Add sequence padding and restoration APIs

Implement `pad_sequence` and `unpad_sequence` under `paddle.nn.utils.rnn`, and expose both operations from `paddle.nn.utils`.

## `pad_sequence`

The API accepts a non-empty list or tuple of Tensors whose first dimension contains sequence length and whose remaining dimensions are compatible.

Required behavior:

- pad every sequence to the maximum input length;
- use time-major output (`T x B x *`) by default;
- use batch-major output (`B x T x *`) when `batch_first=True`;
- pad on the right by default and on the left when requested;
- fill padded positions with the supplied `padding_value`, defaulting to zero;
- preserve input dtype, including integer dtype;
- support a single sequence, equal-length sequences, no trailing dimensions, and multidimensional trailing shapes;
- reject inputs that are not a list or tuple;
- reject padding sides other than `"left"` and `"right"`.

## `unpad_sequence`

The API accepts a padded Tensor, a Tensor containing the original sequence lengths, and the layout selection.

Required behavior:

- return one Tensor per requested length;
- slice the time dimension when `batch_first=False`;
- slice the second dimension when `batch_first=True`;
- preserve trailing dimensions, dtype, and values;
- support single sequences, equal lengths, and multidimensional data.

## Integration And Regression Requirements

Padding followed by unpadding must recover each original sequence for both layouts. Left-padded data must round-trip correctly when the corresponding lengths are supplied. The APIs must be importable from `paddle.nn.utils.rnn` and exposed from `paddle.nn.utils` without regressing existing neural-network utility behavior.
