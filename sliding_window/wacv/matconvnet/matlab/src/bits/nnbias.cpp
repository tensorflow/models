#ifdef ENABLE_GPU
#error "The file nnsubsample.cu should be compiled instead"
#endif
#include "nnbias.cu"

/**
@brief nnbias_forward
@param context context.
@param output output tensor $\by$ [output].
@param outputMult output tensor multiplier $\alpha$.
@param data data tensor $\bx$.
@param dataMult data tensor multiplier $\beta$.
@param biases biases tensor $\bb$.
@param biasesMult biases tensor multiplier $\gamma$.
 
The function computes
@f[
 y_{ijkd} \leftarrow
 \alpha y_{ijkd} +
 \beta x_{ijkd} +
 \gamma b_k.
@f]

@a data can be the null tensor, in which case this tensor
is dropped in the summation.
*/

/**
@brief nnbias_backward
@param context context.
@param derData data derivative tensor $d\bx$ [output].
@param derDataMult data derivative tensor multiplier $\eta$.
@param derBiases biases derivative tensor $d\bb$ [output].
@param derBiasesMult biased derivative tensor multiplier $\tau$.
@param data data tensor $\bx$.
@param dataMult data tensor multiplier $\beta$.
@param biases biases tensor $\bb$.
@param biasesMult biases tensor multiplier $\gamma$.

If @a derData is the null tensor, this derivative is not comptued and
@param biases can also be null.

If @a derBiases is the null tensor, this derivative is not computed and
@param data can also be null.
*/
