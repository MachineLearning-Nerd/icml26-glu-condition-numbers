# Claim 6 source audit

Section 3.3 says that introducing GLU in the FFN blocks of ViT and GPT-2
decreases the condition number. Figure 3 compares ReLU/ReGLU, GELU/GEGLU, and
SiLU/SwiGLU and displays four points per bar.

The official repository at commit
`cc1664bf48505d7bac308b308ce1c495b06ce979` supplies a ViT script with
`N=64,image_size=32,patch=4,dim=4096,depth=4,heads=4,mlp_dim=256`, but supplies
neither GPT-2 model code nor the raw Figure-3 CSV. This reproduction keeps the
official ViT architecture/config, uses the standard GPT-2-small block
dimensions, and discloses the reduced Jacobian sample count (`8` rather than
the official ViT script's `64`).

