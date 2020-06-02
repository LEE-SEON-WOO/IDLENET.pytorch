import os
import sys
try:
    from rulebased_analysis import parse_xml
except ImportError as error:
    import parse_xml
import math
import copy
import numpy as np
import cv2
import xmltodict
import json
import glob
from collections import OrderedDict

def calculateMaxSubmission(y, x, csv_arr):
    submission_arr = []
    if x > 0 and y > 0:
        submission_arr.append(csv_arr[y][x]-csv_arr[y-1][x-1])
    if y > 0:
        submission_arr.append(csv_arr[y][x] - csv_arr[y-1][x])
    if x < len(csv_arr[0])-1 and y>0:
        submission_arr.append(csv_arr[y][x] - csv_arr[y-1][x+1])
    if x > 0:
        submission_arr.append(csv_arr[y][x] - csv_arr[y][x-1])
    if x < len(csv_arr[0])-1:
        submission_arr.append(csv_arr[y][x] - csv_arr[y][x+1])
    if x > 0 and y < len(csv_arr)-1:
        submission_arr.append(csv_arr[y][x] - csv_arr[y+1][x-1])
    if y < len(csv_arr)-1:
        submission_arr.append(csv_arr[y][x] - csv_arr[y+1][x])
    if x < len(csv_arr[0])-1 and y < len(csv_arr)-1:
        submission_arr.append(csv_arr[y][x] - csv_arr[y+1][x+1])

    return max(submission_arr)

def writeJson2(input_data):
    # print('write "{0}" ...'.format(inp))

    data = OrderedDict()
    # xmin
    data["xmin"] = input_data["xmin"]
    input_data.pop("xmin", None)
    # ymin
    data["ymin"] = input_data["ymin"]
    input_data.pop("ymin", None)
    # xmax
    data["xmax"] = input_data["xmax"]
    input_data.pop("xmax", None)
    # ymax
    data["ymax"] = input_data["ymax"]
    input_data.pop("ymax", None)
    # tmin > t_min 2020.04
    data["t_min"] = input_data["tmin"]
    input_data.pop("tmin", None)
    # tmax > t_max 2020.04
    data["t_max"] = input_data["tmax"]
    input_data.pop("tmax", None)
    # tmean
    data["tmean"] = input_data["tmean"]
    input_data.pop("tmean", None)
    # class
    data["class"] = input_data["class"]
    input_data.pop("class", None)

    # class
    #data["confidence"] = input_data["confidence"]
    data["confidence"] = 1.0
    input_data.pop("confidence", None)
    
    # hp
    data["hp"] = []
    hp_contour = input_data["hp_counter"]
    for i in range(len(hp_contour)):
        temp_contour = []
        for j in range(len(hp_contour[i])):
            temp_contour.append('('+str(hp_contour[i][j][0][0]+data["xmin"])+','+str(hp_contour[i][j][0][1]+data["ymin"])+')')
        data["hp"].append(temp_contour)
    input_data.pop("hp_counter", None)

    # rp
    data["rp"] = []
    rp_contour = input_data["rp_counter"]
    for i in range(len(rp_contour)):
        temp_contour = []
        for j in range(len(rp_contour[i])):
            temp_contour.append(
                '(' + str(rp_contour[i][j][0][0] + data["xmin"]) + ',' + str(rp_contour[i][j][0][1] + data["ymin"]) + ')')
        data["rp"].append(temp_contour)
    input_data.pop("rp_counter", None)

    # tp
    data["top_rate"] = []
    tp_counter = input_data["tp_counter"]
    for i in range(len(tp_counter)):
        temp_contour = []
        for j in range(len(tp_counter[i])):
            temp_contour.append('(' + str(tp_counter[i][j][0][0] + data["xmin"]) + ',' + str(
                tp_counter[i][j][0][1] + data["ymin"]) + ')')
        data["top_rate"].append(temp_contour)
    input_data.pop("tp_counter", None)

    # blur_alarm
    # recommend_action
    # image_type = jpg / add 2020.04
    data["image_type"] = "jpg"
    ## Emissivity
    if "Emissivity" in input_data:
        data["Emissivity"] = input_data["Emissivity"]
        input_data.pop("Emissivity", None)
    else:
        data["Emissivity"] = 0.95
    ## Atmospheric Temperature
    if "Atmospheric Temperature" in input_data:
        data["Atmospheric Temperature"] = input_data["Atmospheric Temperature"]
        input_data.pop("Atmospheric Temperature", None)
    else:
        data["Atmospheric Temperature"] = 20
    ## Relative Humidity
    if "Relative Humidity" in input_data:
        data["Relative Humidity"] = input_data["Relative Humidity"]
        input_data.pop("Relative Humidity", None)
    else:
        data["Relative Humidity"] = 40

    ## Point
    if "Point" in input_data:
        data["Point"] = input_data["Point"]
        input_data.pop("Point", None)
    else:
        data["Point"] = "2320-472-M-WV-07PA"

    ## Facilityname > measure_device 2020/04
    if "FacilityName" in input_data:
        data["measure_device"] = input_data["FacilityName"]
        input_data.pop("FacilityName", None)
    else:
        data["measure_device"] = "Fuse"
    ## FileName
    if "FileName" in input_data:
        data["FileName"] = input_data["FileName"]
        input_data.pop("FileName", None)
    else:
        data["FileName"] = "fuse.jpg"
    ## FacilityClass
    if "FacilityClass" in input_data:
        data["FacilityClass"] = input_data["FacilityClass"]
        input_data.pop("FacilityClass", None)
    else:
        data["FacilityClass"] = "ETC"

    ## FacilityClass_option
    if "FacilityClass_option" in input_data:
        data["FacilityClass"] = str(data["FacilityClass"]) + ", " + str(input_data["FacilityClass_option"])
        input_data.pop("FacilityClass_option", None)
    else:
        data["FacilityClass"] = str(data["FacilityClass"]) + ", " + "A상"
    ## Limit Temperature
    if "Limit Temperature" in input_data:
        data["Limit Temperature"] = input_data["Limit Temperature"]
        input_data.pop("Limit Temperature", None)
    else:
        data["Limit Temperature"] = 25
    ## PointTemperature
    if "PointTemperature" in input_data:
        data["PointTemperature"] = input_data["PointTemperature"]
        input_data.pop("PointTemperature", None)
    else:
        data["PointTemperature"] = 35.9
    ## Over temperature
    if "Over temperature" in input_data:
        data["Over temperature"] = input_data["Over temperature"]
        input_data.pop("Over temperature")
    else:
        data["Over temperature"] = "9.9"

    ## Over temperature_option
    if "Over temperature_option" in input_data:
        data["Over temperature_option"] = str(data["Over temperature_option"]) + ", " + str(input_data["Over temperature_option"])
        input_data.pop("Over temperature_option", None)
    else:
        data["Over temperature"] = str(data["Over temperature"]) + ", " + "정상"

    ## deltaTfrom rulebased_analysis import parse_xml
    if "deltaT" in input_data:
        data["deltaT"] = input_data["deltaT"]
        input_data.pop("deltaT", None)
    else:
        data["deltaT"] = 0

    ## Cause of Failure
    if "Cause of Failure" in input_data:
        data["Cause of Failure"] = input_data["Cause of Failure"]
        input_data.pop("Cause of Failure", None)
    else:
        data["Cause of Failure"] = "정상"
    ## DiagnosisCode
    if "DiagnosisCode" in input_data:
        data["DiagnosisCode"] = input_data["DiagnosisCode"]
        input_data.pop("DiagnosisCode", None)
    else:
        data["DiagnosisCode"] = "AA"
    ## Diagnosis
    if "Diagnosis" in input_data:
        data["Diagnosis"] = input_data["Diagnosis"]
        input_data.pop("Diagnosis", None)
    else:
        data["Diagnosis"] = "정상임"
    ## image_size / add 2020.04
    if "image_size" in input_data:
        data["image_size"] = input_data["image_size"]
    else:
        data["image_size"] = "unknown"
    ## blur confidence / add 2020.04
    if "blur_alarm" in input_data:
        data["blur_alarm"] = input_data["blur_alarm"]
    else:
        data["blur_alarm"] = "0"
    print("unused key:")
    for key in input_data.keys():
        print("\t" + key)
    ## recommend_action
    if "recommend_action" in input_data:
        data["recommend_action"] = input_data["recommend_action"]
    else:
        data["recommend_action"] = "연결부 조임"
    return data

def analyze(xml, csv, out_dir, blur_confidence, image_path):
    #xml = Annotation file
    fname = os.path.basename(os.path.splitext(xml)[0])
    analyze = parse_xml.parcingXml(xml)
    analyze = np.array(analyze)
    if csv.endswith('csv'):
        csv_arr = parse_xml.parcingCsv(csv)
    elif csv.endswith('json'):
        with open(csv_path) as f:
            data = json.load(f)   
            csv_arr = np.array(json.loads(data['tempData']))
    file_data = OrderedDict()
    file_data["facilities"] = []

    for i in range(len(analyze[0])):
        print(f"process {i+1} in {len(analyze[0])}")
        xmin = int(analyze[0][i])
        xmax = int(analyze[1][i])
        ymin = int(analyze[2][i])
        ymax = int(analyze[3][i])
        img = cv2.imread(image_path)
        box_img = cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0, 255, 0), 2)
        cv2.imwrite(out_dir+'_'+str(i+1)+'.jpg',box_img)

        object_class = analyze[4][i]
        csv_copy = copy.deepcopy(csv_arr)
        csv_crop = csv_copy[ymin:ymax, xmin:xmax]
        csv_flat = csv_crop.flatten()
        csv_flat = np.round_(csv_flat, 1)
        temp_min = csv_flat.min()
        temp_max = csv_flat.max()
        temp_average = np.average(csv_flat)
        temp_average = np.round_(temp_average, 1)

        # find heating points
        csv_copy = copy.deepcopy(csv_arr)
        csv_crop = csv_copy[ymin:ymax, xmin:xmax]
        thresh = np.percentile(csv_crop, 75)
        thresh_arr = np.zeros((len(csv_crop), len(csv_crop[0])), dtype=np.uint8)
        thresh_arr = np.where(csv_crop[:,:]<thresh, 0 ,255)
        thresh_arr = np.array(thresh_arr, dtype=np.uint8)
        hp_contour, _ = cv2.findContours(thresh_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # find reflection points
        csv_copy = copy.deepcopy(csv_arr)
        csv_crop = csv_copy[ymin:ymax, xmin:xmax]
        CRITICAL_GRAD = 0.4
        thresh = np.percentile(csv_crop, 75)
        thresh_arr = np.zeros((len(csv_crop), len(csv_crop[0])), dtype=np.uint8)
        thresh_arr = np.where(csv_crop[:,:]<thresh, 0, 255)
        thresh_arr = np.array(thresh_arr, dtype=np.uint8)

        # find top rate celsius
        csv_copy = copy.deepcopy(csv_arr)
        csv_crop = csv_copy[ymin:ymax, xmin:xmax]
        thresh = np.percentile(csv_crop, 5)
        thresh_arr = np.zeros((len(csv_crop), len(csv_crop[0])), dtype=np.uint8)
        thresh_arr = np.where(csv_crop[:, :] < thresh, 0, 255)
        thresh_arr = np.array(thresh_arr, dtype=np.uint8)
        tp_contour, _ = cv2.findContours(thresh_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        height, width = thresh_arr.shape
        suspected_points = []
        for j in range(height):
            for k in range(width):
                if thresh_arr[j][k] != 0:
                    temp = calculateMaxSubmission(j, k, csv_crop)
                    if temp > CRITICAL_GRAD:
                        suspected_points.append([k,j])
        
        masking_img = np.zeros((height, width, 3), dtype=np.uint8)
        for pts in suspected_points:
            xy = np.array(pts)
            cv2.circle(masking_img, (xy[0], xy[1]), 3, (255, 255, 255), -1)

        masking_img = masking_img[:,:,0]
        masking_img = masking_img.astype(np.uint8)
        rp_contour, heirachy = cv2.findContours(masking_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        
        # object_data = writeJason(xmin, ymin, xmax, ymax, temp_min, temp_max, temp_average, object_class, hp_contour, rp_contour)
        json_data = {}
        json_data["xmin"] = xmin
        json_data["ymin"] = ymin
        json_data["xmax"] = xmax
        json_data["ymax"] = ymax
        json_data["tmin"] = temp_min
        json_data["tmax"] = temp_max
        json_data["tmean"] = temp_average
        json_data["class"] = object_class
        json_data["hp_counter"] = hp_contour
        json_data["rp_counter"] = rp_contour
        json_data["tp_counter"] = tp_contour
        json_data["image_size"] = str(csv_arr.shape[1])+"X"+str(csv_arr.shape[0])
        json_data["FileName"] = csv.split(os.sep)[-1]
        rule = DiagnosisRule("./data/diagnosis_rule.json")
        rule_dict = rule.diagnose(object_class, temp_average)
        json_data['FacilityName'] = rule_dict['name']
        json_data["Limit Temperature"] = rule_dict["Limit Temperature"]
        json_data["blur_alarm"] = str(blur_confidence)
        file_data["facilities"].append(writeJson2(json_data))

        # file_data["facilities"].append(object_data)
        # with open('./json_rb/'+fname+'.json', 'w', encoding='utf-8') as make_file:
        with open(out_dir+'_'+str(i+1)+'.json', 'w', encoding='utf-8') as make_file:
            json.dump(file_data, make_file, indent="\t", ensure_ascii=False)

class DiagnosisRule():
    def __init__(self, json_filename):
        json_file = open(json_filename, encoding='utf-8')
        self.rule_data = json.load(json_file)
            
    def diagnose(self, f_class, temperature, rule="ITC"):
        if f_class not in self.rule_data[rule].keys():
            f_class_original = f_class
            f_class = "undefined"
        
        name = self.rule_data[rule][f_class]["name"]
        LimitTemp = self.rule_data[rule][f_class]["LimitTemp"]
        
        fail_return = {"code":"NR", "cause":"정상", "action":"정상임", "Over Temperature":0, "name":name, "Limit Temperature": LimitTemp }
        

        for fail in self.rule_data[rule][f_class]["failure"]:
            # print(fail)
            if temperature >= LimitTemp + fail["dT"]:
                fail_return = fail
                fail_return.pop("dT", None)
                fail_return["Over Temperature"] = temperature - LimitTemp
                fail_return["name"] = name
                fail_return["Limit Temperature"] = LimitTemp            
        return fail_return
        
# if __name__=="__main__":
#     #example
#     data = os.path.join('D:\\2020연구\\1) 한수원\\2분기\\20.04 우선구현 파일(이노팩토리)\\회전설비')
#     xml_folder_path =  os.path.join(data,'annotations')
#     csv_folder_path = os.print('Arguments:')path.join(data,'json')
    
#     xml_folder = os.listdir(xml_folder_path)
#     csv_folder = os.listdir(csv_folder_path)
    
    
#     out_dir = os.path.join("D:\\2020연구\\1) 한수원\\2분기\\20.04 우선구현 파일(이노팩토리)\\회전설비", "results")
#     for xml, csv in list(zip(xml_folder, csv_folder)):
#         xml_path = os.path.join(xml_folder_path, xml)
#         csv_path = os.path.join(csv_folder_path, csv)
#         print(xml_path, csv_path)

#         analyze(xml_path, csv_path, out_dir)

