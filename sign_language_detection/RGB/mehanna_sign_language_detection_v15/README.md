# Version 15

- In this version we changed the way of geting the frames to discard the part of data preparing in the previous versions
    - In the previous versions we was extracting all of the frames and write them on the hard-disk then when we need to get video's frames we read them from the hard-disk
    - Now we discard that part and we only read the video in the runtime then return its frames without writing them into the hard-disk

- Also in this version we tried to classify 100 signs

- After 30 epoch training on my device we got ~65% training accuracy and ~3% validation accuracy
- After 60 epoch training on colab we got ~80% training accuracy and ~4% validation accuracy