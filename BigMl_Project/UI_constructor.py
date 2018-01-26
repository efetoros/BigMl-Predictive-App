pickle_store = ''

import os
os.environ['BIGML_USERNAME'] = "efetoros"
os.environ['BIGML_API_KEY'] = "471ae5485d74ceeb2e911e0c1d37edda58cf79d3"
import bigml
from bigml.ensemble import Ensemble
from bigml.model import Model
from bigml.api import BigML
from tkinter import *
import pickle

API = BigML()
f = open(pickle_store, 'rb')
obj = pickle.load(f)
feature_names = obj[0]
model_or_ensemble = obj[1]
local_MoE = obj[2]
f.close()


def predictor(input_data):
  return local_MoE.predict(input_data, add_confidence=True)
  

window = Tk()
window.title("Prdective App")
window.configure(background="dark grey")

def click():
  collector = {}
  output.delete(0,END)
  print("check 1")
  for i in range(len(features)):
    x= feature_names[i]
    y= inputs[i].get()
    collector[x]=y
  print(collector)
  answer= predictor(collector)
  try:
    answer= predictor(collector)
  except: 
    answer="Inputs are incorrect"
  output.insert(END,answer)

def click2():
  output_batch.delete(0,END)
  e = entry_batch.get()
  try:
    source = API.create_source(e)
    dataset = API.create_dataset(source)
    batch_prediction = API.create_batch_prediction(model_or_ensemble, dataset,{"all_fields": True})
    API.ok(batch_prediction)
    API.download_batch_prediction(batch_prediction,
                              filename= (e[:-4]+ "_Batch_Prediction.csv"))
    answer= "DOWNLOADED"
  except: 
    answer= "INVALID PATH"
  output_batch.insert(END,answer)


#photo
photo1 = PhotoImage(file="Webp.net-gifmaker.gif")
photo2 = PhotoImage(file="process_dataset.gif")

Label(window, image=photo1, bg="grey").grid(row=0, column=1, sticky=W)
Label(window,height=300, image=photo2, bg="grey").grid(row=1, column=0,rowspan=10, sticky=W)

Label(window, text="""  Predictive 
  Application 
 """, bg="dark grey",width=17,height=10,font=("Courier", 30)).grid(row=0, column=0, sticky=W)

global inputs
global features
inputs = []
features = []
for i in range(1, len(feature_names)+1):
  num = i
  i = Entry(window,width=40,bg="white")
  i.grid(row=num,column=1,sticky=S)
  inputs.append(i)

num = 1
for item in feature_names:
  name = item
  item = Label(window, text= name + ":", bg="dark grey").grid(row=num, column=1, sticky=W)
  features.append(item)
  num = num + 1

Button(window,text="Predict",width=7,command=click).grid(row=len(feature_names)+1,column=1,sticky=S)

output_label= Label(window, text="PREDICTION:",font=("Courier",14), bg="dark grey").grid(row=len(feature_names)+2, column=1, sticky=W)
global output
output= Entry(window, width= 60, bg="white")
output.grid(row=len(feature_names)+2,column=1,sticky= S)



Label(window, text="""BATCH PREDICTION
FILE-PATH""",font=("Courier",12), bg="dark grey").grid(row=11, column=0, sticky=W)
global entry
entry_batch = Entry(window,width= 28, bg="white")
entry_batch.grid(row=11,column=0,sticky= E)

Button(window,text="Predict",width=7,command=click2).grid(row=12,column=0,sticky=S)
global output_batch
output_batch = Entry(window, width= 17, bg="white")
output_batch.grid(row=12,column=0,sticky= E)




window.mainloop()



