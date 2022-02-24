# V1

- ensemble pytorch conv model with and keras LSTM model normalize the pytorch output then sum both predections
-        Keypoits    RGB     ensemble
-  val   95.2        93.6    96.84
-  test  91.6        89.8    95.83



- using real time with two models has some problems 
    - memory limit of the gpu
    - very slow model
    - failed to run one model on cpu and the other on GPU
    