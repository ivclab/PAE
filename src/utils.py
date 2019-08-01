import os

class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)

def get_dataset(path, has_class_directories=True):
  dataset = []
  path_exp = os.path.expanduser(path)
  classes = [path for path in os.listdir(path_exp) \
                  if os.path.isdir(os.path.join(path_exp, path))]

  classes.sort()
  nrof_classes = len(classes)
  for i in range(nrof_classes):
    class_name = classes[i]
    target_dir = os.path.join(path_exp, class_name)
    image_paths = get_image_paths(target_dir)
    dataset.append(ImageClass(class_name, image_paths))

  return dataset

def get_image_paths(target_dir):
  image_paths = []
  if os.path.isdir(target_dir):
    images = os.listdir(target_dir)
    image_paths = [os.path.join(target_dir, img) for img in images]
  return image_paths