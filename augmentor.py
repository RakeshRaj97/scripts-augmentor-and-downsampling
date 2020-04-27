import Augmentor

p =  Augmentor.Pipeline("path")
p.crop_by_size(probability=1, width=200, height=200, centre=False)
p.sample(2000)
