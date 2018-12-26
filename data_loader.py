# Utils
# Setting
from yolo_config import *

base_path = os.getcwd()
sys.path.append(base_path)

# txt annotation 파일
path = os.path.join(base_path, 'data/annotation/*.txt')
def findFiles(path): return glob.glob(path)
filenames = findFiles(path)

# 파일을 읽고 줄 단위로 분리
def readLines(filename):
    lines = ' '.join(open(filename, encoding='utf-8').read().strip().split('\n'))
    return lines


GT_bboxes = []
for filename in filenames:
    lines = readLines(filename)
    lines = [float(string) for string in lines.split()]
    arr = np.array(lines)
    GT_tensor = torch.from_numpy(arr)
    GT_bboxes.append(GT_tensor)

print(GT_bboxes)




# 이미지 파일
# transform.CenterCrop(416)
data_transform = transforms.Compose([transforms.Resize(416),
                                     transforms.CenterCrop(416),
                                     transforms.ToTensor()
                                     ])

dataset = datasets.ImageFolder(root=os.path.join(base_path, 'data/train'),
                                           transform=data_transform)

loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

for i, data in enumerate(loader):
    print(data[0].size())  # input image
    print(data[1])         # class label








