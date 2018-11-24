This repo contains a collection of sequential prediction model implementations in a few different languages/libraries,
mostly sandbox stuff. The interfaces may be cleaned up in the future, but these projects are mostly a research sandbox.

Models and implementations:
1) Vanilla rnn in numpy
2) Vanilla rnn in torch, manually constructed (not builtin torch.nn.RNN module)
3) Torch gru implementation for one-hot encoded symbol sequences


./resources: junk folder for images, etc.
./models: saved parameters
./ : duh code
./data: training data

Training data is Treasure Island, by Robert Louis Stevenson.

![gru error](resources/pytorch_gru_error.png)

Gru character generation after modest training:
```

```
