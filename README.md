# Building Neural Networks

We love NNs :D

I only knew how they worked in theory
I was super stuck on backProp for some reason
So I thought I'll make this bad boy :P

## Implementing Gradient Descents via OOPS:
yoo, if you want to do that, check Basic

## Implementing Gradient Descents via Matrices:
oo, you are big boy/girl. In that case, check MatrixANN.

Also check the Results folder to see how my model did :D

Although it is a super simple task, my model took and awful lot of iterations. I will be the bigger guy here and BLAME EVERYTHING ON RANDN initialization >:C

|Model|error_00|error_01|error_10|error_11|mean_absolute_error|iterations|learning_rate|time|
|-----|--------|--------|--------|--------|-------------------|----------|-------------|----|
|Treating weights as matrices with Randn|-0.05893545088738524|0.056224441450036355|0.05176583901218468|-0.055152695839147274|0.05551960679718839|20000|0.08|0.53s|
|Treating a Neuron Like a Class and weights as a class property lowk|-0.01756189411514389|0.025442847242652333|0.013488335667694806|-0.02427675441307189|0.02019245785964073|20000|0.08|1.24s|

WHAT, my favorite child isnt doing as good as the other one... hmmmmmm
Well I'd like to be the bigger person and BLAME RANDN AGAIN omfg. (Dw the fav child performs better once you crank up the alphas)
Favorite child is roughly 60% faster btw.

### Hey Tanvay, ur super cool and all but what's randn?
Glad you asked!

randn is basically a normal distribution. One of the bell curves you might have seen
Theoretically it can lie from -inf to inf (slang for infinity)
But the probability of the values being close to 0 are super high.

So most of my weights are super small while initialization. Instead of doing np.random.randn, I will do .rand
This uniformly distributes from 0 to 1.

Such a simple problem shouldnt take more than 1-2k iterations frankly so 20k is super high ngl