from video_features.i3dmodel import i3dmodel
from maf.openvadmodel import openVAD
from maf.option import parser
import matplotlib.pyplot as plt

args = parser.parse_args()

path = 'C:/Users/christ/Downloads/OpenVAD_MAF/OpenVAD_MAF/sample1.mp4'

feature = i3dmodel(path)

openvad = openVAD(args)

threshold = 0.62

features = feature.forward()
output = openvad.forward(features)
top_three = sorted(output, reverse=True)[:3]
count = 0
for i in top_three:
    if i > threshold:
        count += 1
if count ==3:
    print("Anomaly")
else:
    print("Normal")
# print(output)