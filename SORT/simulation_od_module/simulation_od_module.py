import os 
import sys 

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())   

from typing import Dict, List 
from utils.general import read_yaml_file
import pandas as pd 
from pathlib import Path 



class SimulationOD():

    def __init__(self, path_to_scenario_folder:Path) -> None:
        self.images_files_path = path_to_scenario_folder / 'img1'
        self.gt_file_path = path_to_scenario_folder /'gt'/ 'gt.txt'
        self.detection_file_path = path_to_scenario_folder /'det'/ 'det.txt'

        # Create and define the object detection module output (GT and Estimated)
        self.columns_gt=  ['frame', 'id', 'x', 'y', 'w', 'h', 'person/not', '?', '?']
        self.od_gt_df = self.load_files(self.gt_file_path, self.columns_gt)
        self.od_df = self.load_files(self.detection_file_path, columns= self.columns_gt[:7])
        self.images_names =  os.listdir(self.images_files_path)

        # Cretae a dataframe for the images path 
        self.images_dict = {}
        for i in range(len(self.images_names)):
            image_name = self.images_names[i].split('.')[0]
            self.images_dict[int(image_name)-1] = os.path.join(self.images_files_path, self.images_names[i])

        # Scene length 
        self.duration = len(self.images_dict) 


    def reset(self):
        self.t = 1

    def __len__(self):
        return self.duration


    def load_files(self,file:Path,  columns: List[str] = None) -> pd.DataFrame:
        with open(file) as file:
            data = file.readlines()
        
        od_data = []
        for i in range(len(data)):
            raw_data = data[i].replace('\n','')
            od_data += [raw_data.split(',')]
        
        if columns is None:
            columns = range(len(od_data[0]))

        od_data = pd.DataFrame(columns = columns, data=od_data)
        od_data=od_data.astype(float)

        od_data['frame'] = od_data['frame'].astype(int) - 1
        return od_data
      
    def __getitem__(self, frame_idx)->Dict:
        
        if self.t != self.duration: 
            done = False
            state =  self.od_df[self.od_df['frame'] == frame_idx]
            frame_img_path = self.images_dict.get(frame_idx, None)
            od_gt = self.od_gt_df[self.od_gt_df['frame'] == frame_idx]
        else:
            done = True
            state =  None
            frame_img_path = None
            od_gt = None

        observation_ang_gt = {
            'state':state,
            'od_gt': od_gt,
            'frame_img_path': frame_img_path,
            "done": done 
        }


        self.t += 1
        return observation_ang_gt




# class ODModelSimulation:
#     def __init__(self, config: Dict)->None:
#         config_od_model = config['object_detection_model']
#         path_to_data_set = config['path_to_data_set']   
#         self.od_model = config_od_model['object_detection_model']['name']


#         self.train_fodler = os.path.join(path_to_data_set, 'train')
#         self.test_folder = os.path.join(path_to_data_set, 'test')

        
#         self.mot_scenarios = {}

#         list_of_trained_folder = os.listdir(self.train_fodler)
#         list_of_test_folder = os.listdir(self.test_folder)

#         self.train_folder_scenarios = [folder_name for folder_name in list_of_trained_folder if self.od_model in folder_name]
#         self.test_fodler_scenarios = [folder_name for folder_name in list_of_test_folder if self.od_model in folder_name]


#     def define_all_scenarios(self) -> Dict[str, SimulationOD]:
#         pass 


    
if __name__ == "__main__":
    path_to_scenario_folder = Path(r"C:\Projects\datasets\MOT17\MOT17_dataset\train\MOT17-02-FRCNN")
    simulation_od  = SimulationOD(path_to_scenario_folder = path_to_scenario_folder)
    simulation_od.reset()
    N = len(simulation_od)

    for i in range(N):
        state = simulation_od[i]



