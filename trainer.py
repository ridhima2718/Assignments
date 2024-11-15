import os.path
import datetime
import cv2
import numpy as np
from core.utils import preprocess, metrics
import torch
from function import *
import tensorflow as tf
import json

def train(model, ims, real_input_flag, configs, itr):
    model.train()
    ims_tensor = torch.tensor(ims, dtype=torch.float32).to(configs.device)
    real_input_flag_tensor = torch.tensor(real_input_flag, dtype=torch.float32).to(configs.device)

    _, cost = model(ims_tensor, real_input_flag_tensor)
    
    if configs.reverse_input:
        ims_rev = np.flip(ims, axis=1).copy()
        ims_rev_tensor = torch.tensor(ims_rev, dtype=torch.float32).to(configs.device)
        _, cost_rev = model(ims_rev_tensor, real_input_flag_tensor)
        cost = (cost + cost_rev) / 2

    if itr % configs.display_interval == 0:
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Iteration:', itr)
        print('Training Loss:', cost.item())

    if itr % configs.save_interval == 0:
        model_path = 'epsilon_model.h5'
        config_path = 'epsilon_model.json'
        
        torch.save(model.state_dict(), model_path)
        print(f"Model weights saved to {model_path}")
        
        model_config = {
            'num_layers': model.num_layers,
            'num_hidden': model.num_hidden,
            'configs': model.configs.__dict__,
        }
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=4)
        print(f"Model configuration saved to {config_path}")

    return cost.item()



def test(model):
    model_from_json = tf.keras.models.model_from_json
    
    json_file = open("epsilon_model.json", "r")
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights("epsilon_model.h5")

    colors = []
    for i in range(0,20):
        colors.append((245,117,16))
    print(len(colors))
    

    sequence = []
    sentence = []
    accuracy=[]
    acc_list=[]
    predictions = []
    threshold = 0.8 

    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():

            ret, frame = cap.read()
            cropframe=frame[40:400,0:300]
            frame=cv2.rectangle(frame,(0,40),(300,400),255,2)
            image, results = mediapipe_detection(cropframe, hands)
            
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            try: 
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(actions[np.argmax(res)])
                    predictions.append(np.argmax(res))
                    
                    
                #3. Viz logic
                    if np.unique(predictions[-10:])[0]==np.argmax(res): 
                        if res[np.argmax(res)] > threshold: 
                            if len(sentence) > 0: 
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                                    accuracy.append(str(res[np.argmax(res)]*100))
                            else:
                                sentence.append(actions[np.argmax(res)])
                                accuracy.append(str(res[np.argmax(res)]*100)) 

                    if len(sentence) > 1: 
                        sentence = sentence[-1:]
                        accuracy=accuracy[-1:]

            except Exception as e:
                pass
                
            cv2.rectangle(frame, (0,0), (300, 40), (245, 117, 16), -1)
            cv2.putText(frame,"Character: "+' '.join(sentence)+" Acc: "+''.join(accuracy), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('OpenCV Feed', frame)
            if accuracy:
                acc_list.append(float(accuracy[0]))
            if cv2.waitKey(10) & 0xFF == ord('q'):
                print('Accuracy: ',np.mean(acc_list))
                break
        cap.release()
        cv2.destroyAllWindows()

