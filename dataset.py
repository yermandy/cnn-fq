import os
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from warnings import filterwarnings

ImageFile.LOAD_TRUNCATED_IMAGES = True
filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


def join(*paths):
    return os.path.join(os.path.dirname(__file__), *paths)


class ListDataset(Dataset):
    def __init__(self,
                images,
                bounding_boxes,
                labels=None,
                scale=0.5, 
                transform=None,
                path_to_images=None,
                weights=None):
        self.images = images
        self.bbs = bounding_boxes
        self.labels = labels
        self.weights = weights
        self.scale = scale
        self.path_to_images = path_to_images if path_to_images is not None else 'images'
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform


    def __getitem__(self, index):
        path  = self.images[index]
        bb    = self.bbs[index]
        image = Image.open(join(self.path_to_images, path))
        image = image.convert('RGB')
        image = self.crop_face(image, bb)
        image = self.transform(image)
        if self.weights is not None:
            alpha = self.weights[index]
            return (image, alpha)
        else:
            return image
        

    def __len__(self):
        return len(self.images)


    def crop_face(self, img, bb):
        x1, y1, x2, y2 = bb
        w_scale = ((x2 - x1) * self.scale) / 2
        h_scale = ((y2 - y1) * self.scale) / 2
        x1 -= int(w_scale)
        y1 -= int(h_scale)
        x2 += int(w_scale)
        y2 += int(h_scale)
        return img.crop((x1, y1, x2, y2))


    def get_labels(self):
        return self.labels