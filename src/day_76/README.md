## Day 76
Adding the first step of MHA in cutlass. We'd like to perform:

$
scores = softmax(QK^T)
$

Note: In cutlass we don't need to physically transpose K, but we might need to add a custom epilogue for softmax. We'd ideally like to do this using online softmax.