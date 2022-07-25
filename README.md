# PyTorch-SimMIM
## Description
This is an implementation of SimMIM, a simple framework for masked image modelling, in PyTorch. You can find the accompanying blog [here](https://borna-ahz.medium.com/simmim-in-pytorch-64fdde781d5b).

## Usage
```SimMIM``` in ```model.py``` is the central class of this implementation, and its behaviour is straightforward. It receives a vision transformer from the ```timm``` library, 
as well as an optional masking ratio, and in the forward pass applies the SimMIM recipe using the provided vision transformer and masking ratio. Its return value
is a tuple containing the number of tokens that were masked, the original values of the patches that were masked, and their reconstructed versions. 

For more information, please view the code and the accomanying docstrings.

## Example

Below, a ```timm``` ViT-Small is trained with the AdamW optimizer for 100 epochs. ```dataloader``` must simply 
fetch images with no labels or annotations, and the only necessary transform is normalization, albeit basic augmentations like
random horizontal flipping and colour jittering help.

```python3
from timm import (
	create_model,
	)
from torch.nn.functional import (
	l1_loss,
	)
from torch.optim import (
	AdamW,
	)


n_epochs = 100
vit = create_model(
	'vit_small_patch32_224',
	num_classes=0,
	)
simmim = SimMIM(
	vit=vit,
	masking_ratio=0.5,
	)
optimizer = AdamW(
	params=simmim.parameters(),
	lr=1e-4,
	weight_decay=5e-2,
	)

for epoch in range(n_epochs):
	for input in dataloader:
		n_masked_tokens, masked_patches_reconstructed, masked_patches_original = simmim(input)

		loss = l1_loss(
			input=masked_patches_reconstructed,
			target=maskes_patches_original,
		)
		loss /= n_masked_tokens
		loss.backward()
		
		optimizer.backward()
		optimizer.zero_grad()
```
