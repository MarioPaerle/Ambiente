The objective of these ideas, are Arc Agi 1/2 and even 3 in future.

## Premise
The Arc Agi are a group of very difficult-for-ai challenges, which are oriented to competitive research.

The Arc 1 and Arc 2 are static in-out example, where small colored grids are linked by a logical rule, often pretty close to human priors, which has to be discovered in order to infer what the output is for the last input given.

Is a very-few shot task, where you need to predict, the output for a given input, without around 2/3 examples.
The fact is that to search for the rule, the human brain usually performs a sort of internal combinatorial search over ton of memories, and since puzzles are human created, and humans share a lot of priors and experience, (their brain "talk the same language") for them is wayy easier than for AIs.

The first difficult barrier is the lack of training data, and since supervised Deep Learning relies heavily on huge amounts of training data, its difficult to train models to do this job, and the second constrain is efficiency: since competition submission are evaluated on 12 GPU kaggle hours, for Arc Agi 2 this means 12 4xL4 hours.


# My first Idea
This said, all the problem seem to reduce simply to: **find the right rule**
yet finding the rule for an AI is pretty difficult, many have tried searching the rule over finite discrete spaces, by creating a Domain Specific Language, and performing discrete search over it, often guided by LLMs.
To finetune this LLMs, usually they use a big ammount of pre-training, LLM-Generated datas.

My idea is to completely change paradigma, and leaping into the continuous.
I would like to have models, that share an embedding space conscious of what objects are, what physics is, trying to make models aware of what acting means.

## NACT
This is just a cool name for this simple concept.
I would like to train a model to operate on a continuous latent space where operation is defined.
NACT1 is all about learning to generate the applied transformation on a grid, given the transformation, where a transformation is the sum (nact) or concatenation (nart) of learned embeddings which is/are given as conditioning to a discrete diffusion model.
Discrete since we're talking about 9 colors -> a very discrete-ish problem with a small vocab. (this enables mask-git style generation)
That said, this model enables us a soft conditioning, continuous, and thus solves the problem of non expressible operation given a DSL.

No one could simply try to train a second model, to use the first one 
# My Second Idea: Nacturn 2
This one idea, is the evolution, of the first one, I designed it anaware that I was ending up pretty close to what Jepa and dino v3 are about.
To teach the concept of "objectness", "presence", etc... I wouldl like to create a model which is able to understand that if I rotate an object of 4 on a grid, that grid is still the same thing, yet with a small variation.
I divided this concept idea in 3 modalities, ranging from the easiest one to the hardest:
#### Mode 1:
This modality is intended to be more about trying to understand if everything could work instead of being really useful.
You simply create an AutoEncoder, probably in ViT style, with a vectorial latent, then I'll encode both a grid, and a small variation of it (crop, hole, rotation, changing of color, etc...), then I'll ask a decoder to reconstruct both of them (or maybe the original one only)
Then I'll add a term to the loss, which makes so that the latent of same varied grids are similar. Of course one can extend this to Contrastive Learning, by making the model separate latents of completely different originated grids.
$$\mathcal{L} = -\alpha \ sim(\hat{A}, \hat{B}) + recon(A,B)$$
So what I would like to find is a very organized and informative latent, and a pretty intelligent model, whose attention maps are enough to see the segmentation it learnt.
To precise, I expect to use Cross Entropy as reconstruction loss, since we're talking about a small vocabulary.

#### Mode 2
This second modality, which i would use in the case the first one turns out to be working is probably a little more useful, but has some problems, I'll try to resolve in the third part:
The idea is to simply pass to the encoders both 2 example and an operation vector, from a finite learned vocabulary. For example:
encode(A, B, rot90). This now expand the similarity loss from before, adding a term: $$\mathcal{L} = -\alpha \ sim(\hat{A}, \hat{B}) + \beta \ ||\hat{B} - \hat{A} - T|| + recon(A,B)$$
thus pushing towards a latent constructed around $\hat{B} \approx \hat{A} + T$
The reconstruction loss still need to be there to prevent collapse... 

Of course this has a major flaw: we're constricting operations to be a linear addition, which might not be real at all. 
For now lets accept the linearity, this enables doing this in inference:
after being given an example (when more we simply average), we can encode both the input and output, then try to derive the T operation, as simply as $T = \hat{B} - \hat{A}$, and to solve the linearity problem, one could think about using small MLPs, trained at Test Time, to the objective: $\hat{B} = MLP(\hat{A})$.
Then given the final input question, we can encode it, apply our MLP trained at test time, and then retrieve the output, by decoding.
Of course test, time, training will probably be not enough, and to better augment the expressivness of the model , I thought we could simply train a secondary model from scratch on NVARC dataset and other Arc Agi training things, to literally generate the T Hypernetwork MLP, in some way, or to simply generate a string of Ts in autoregressive manner, then applying them sequenctially to the A: $\hat{B} = T_{1}(T_{2}(T_{3}\dots T_{n}(A)))$

#### Mode 3
This is simply an extension of what mode 2 does but it trains directly with small MLPs operation creating an embedding of operation which is not simply a list of vector, but a list of MLPs.


# Details
This model should not solve the tasks directly, but indeed it should be used to create a very informative latent, or better, to make sure, we have a model that sees like a human, producing continuous encodings, with high representation power!


### Architecture, Infrastructure & Experiment design
To see if this works, I'll need to make incremental difficulty experiments, and most of all understand incrementally, which architecture and loss fits the problem well.

### Environment
The first thing I'll need to design is an environment capable of easily generating various type of grids and apply several diverse operation to them: from rotations, to color scheme changes, to single object level mutations, or gravity, laser, reflection, and similar things.
I need an easy to use structured, python native environment which is higly efficient, yet has an easy graphical API which I can use to tune the operations, and play with the grid coding system.
It needs to be oriented to pytorch tensors, to make data generation easier for this scope, for example naively supporting various operations, which are both realted or not to my work. 
For example it needs to be super easy and documented how to modify and add new operations as python method, to be able to apply them both with native python or efficient method such as numpy or pytorch.
An idea of the use cases, are augmenting dataset on the fly, in the dataloader itself, by defining a list of operation etc etc...
The scheme and the code have to be incredibly fluent and non complex, for human to read and modify, well organized, and with examples.


#### Objective 1
The first objective is to understand whether a simple contrastive encoder is enough to understand if a grid, and the same grid with a simple operation made are easily understood by the model to be the almost the same high level object.
No decoding, just pure CLIP style learning.The architecture I would start from for this task is a simple Convolutional encoder, + some linear layers to a vector BottleNeck, and later a ViT taking the CLS.
I would like to implement cool visualization methods to be able to see what my model is "thinking" especially with the ViT one, seeing the most related patches with a given one, to see if automatically learns to segment!
