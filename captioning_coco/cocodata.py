import torch
from PIL import Image

class COCO(torch.utils.data.Dataset):
  def __init__(self, img_paths, captions, transforms, tokenizer):
      self.img_paths = img_paths
      self.captions = tokenizer(captions, padding = True, max_length = 20, truncation = True, return_tensors = 'pt')['input_ids']
      self.transforms = transforms

  def __len__(self):
      return len(self.captions)

  def __getitem__(self, index):
      image = self.img_paths[index]
      image = Image.open(image)
      image = self.transforms(image)

      targets = self.captions[index]
      return image, targets
