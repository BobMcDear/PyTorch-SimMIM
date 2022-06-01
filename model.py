from typing import (
	Tuple,
	)

from einops import (
	rearrange,
	)
from timm.models.vision_transformer import (
	VisionTransformer,
	)
from torch import (
	Tensor,
	)
from torch.nn import (
	Linear,
	Module,
	Parameter,
	)
from torch import (
	randn,
	zeros,
	)


def get_indices_to_mask(
	batch_size: int,
	n_tokens: int,
	n_masked_tokens: int,
	device: str = 'cuda',
	) -> Tensor:
	"""
	Gets a set of indices per row for masking
	
	Args:
		batch_size (int): Batch size
		n_tokens (int): Number of tokens per row
		n_masked_tokens (int): Number of tokens to mask per row
		device (str): Desired device for the indices.
		Default is 'cuda'
	
	Returns (Tensor): Set of indices per row for masking
	"""
	indices_to_mask = randn(batch_size, n_tokens, device=device)
	indices_to_mask = indices_to_mask.topk(
		k=n_masked_tokens,
		dim=1,
		)
	indices_to_mask = indices_to_mask.indices
	return indices_to_mask


def get_bitmask(
	batch_size: int,
	n_tokens: int,
	n_masked_tokens: int,
	device: str = 'cuda',
	) -> Tensor:
	"""
	Gets a bitmask for masking

	Args:
		batch_size (int): Batch size
		n_tokens (int): Number of tokens per row
		n_masked_tokens (int): Number of tokens to mask per row
		device (str): Desired device for the bitmask.
		Default is 'cuda'
	
	Returns (Tensor): Boolean tensor with True corresponding to masking 
	the associated token
	"""
	indices_to_mask = get_indices_to_mask(
		batch_size=batch_size,
		n_tokens=n_tokens,
		n_masked_tokens=n_masked_tokens,
		device=device,
		)
	
	bitmask = zeros(batch_size, n_tokens, device=device)
	bitmask = bitmask.scatter(
		dim=1,
		index=indices_to_mask,
		value=1,
		)
	bitmask = bitmask.bool()
	return bitmask


def do_mask_tokens(
	tokens: Tensor,
	mask_tokens: Tensor,
	bitmask: Tensor,
	) -> Tensor:
	"""
	Masks the tokens with a mask token given a bitmask

	Args:
		tokens (Tensor): Tokens to mask
		mask_tokens (Tensor): Tensor with the same shape as tokens filled with
		a mask token
		bitmask (Tensor): Bitmask for masking

	Returns (Tensor): The tokens masked with mask_tokens where bitmask is 
	True 
	"""
	bitmask = bitmask.unsqueeze(2)
	tokens = (~bitmask)*tokens + bitmask*mask_tokens
	return tokens


def get_patches(
	input: Tensor,
	patch_height: int,
	patch_width: int,
	) -> Tensor:
	"""
	Gets patches from input

	Args:
		input (Tensor): Input
		patch_height (int): Patch height
		patch_width (int): Patch width
	
	Returns (Tensor): Patches of the input
	"""
	pattern = (
	'batch_size n_channels (n_patches_height patch_height) (n_patches_width patch_width) -> '
	'batch_size (n_patches_height n_patches_width) (n_channels patch_height patch_width)'
  	)

	patches = rearrange(
		tensor=input,
		pattern=pattern,
		patch_height=patch_height,
		patch_width=patch_width,
		)
	return patches


def get_masked_patches_original(
	input: Tensor,
	patch_height: int,
	patch_width: int,
	bitmask: Tensor, 
	) -> Tensor:
	"""
	Gets patches from input that are supposed to be masked

	Args:
		input (Tensor): Input to extract patches from
		patch_height (int): Patch height
		patch_width (int): Patch width
		bitmask (Tensor): Bitmask that was used for masking
	
	Returns (Tensor): Original version of the patches that are supposed
	to be masked
	"""
	patches = get_patches(
			input=input,
			patch_height=patch_height,
			patch_width=patch_width,
			)
	maskes_patches_original = patches[bitmask]
	return maskes_patches_original


class SimMIM(Module):
	"""
	SimMIM
	"""
	def __init__(
		self,
		vit: VisionTransformer,
		masking_ratio: float = 0.5,
		) -> None:
		"""
		Sets up the modules

		Args:
			vit (VisionTransformer): timm ViT to train
		"""
		super().__init__()

		self.vit = vit
		self.patch_height = vit.patch_embed.patch_size[0]
		self.patch_width = vit.patch_embed.patch_size[1]
		self.n_tokens = vit.patch_embed.num_patches
		self.n_masked_tokens = int(masking_ratio*self.n_tokens)

		self.mask_token = Parameter(randn(vit.embed_dim))
		decoder_out_dim = 3*self.patch_height*self.patch_width
		self.decoder = Linear(
			in_features=vit.embed_dim,
			out_features=decoder_out_dim,
			)

	def forward(
		self,
		input: Tensor,
		) -> Tuple[int, Tensor, Tensor]:
		"""
		Performs a SimMIM forward pass

		Args:
			input (Tensor): Input
		
		Returns (Tuple[int, Tensor, Tensor]): Tuple containing the number of
		masked tokens, the original version of the patches that were masked,
		and the reconstructed version of the patches that were masked
		"""
		batch_size, n_channels, height, width = input.shape
		device = input.device

		tokens = self.vit.patch_embed(input)
		mask_tokens = self.mask_token.repeat(batch_size, self.n_tokens, 1)

		bitmask = get_bitmask(
			batch_size=batch_size,
			n_tokens=self.n_tokens,
			n_masked_tokens=self.n_masked_tokens,
			device=device,
			)
		
		tokens = do_mask_tokens(
			tokens=tokens,
			mask_tokens=mask_tokens,
			bitmask=bitmask,
			)
		
		tokens = tokens+self.vit.pos_embed[:, 1:]
		encoded = self.vit.blocks(tokens)

		masked_tokens_encoded = encoded[bitmask]
		masked_patches_reconstructed = self.decoder(masked_tokens_encoded)
		
		masked_patches_original = get_masked_patches_original(
			input=input,
			patch_height=self.patch_height,
			patch_width=self.patch_width,
			bitmask=bitmask,
			)

		return self.n_masked_tokens, masked_patches_reconstructed, masked_patches_original
		