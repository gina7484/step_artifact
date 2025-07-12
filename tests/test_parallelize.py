"""
Static Parallelization

[B//32, 1] (tile: [32, 128])

RetileStreamify(split_row=True, filter_mask=True)

[B, 1] (tile: [1,128])

Parallelize(ranke = 1, factor=8)
"""
