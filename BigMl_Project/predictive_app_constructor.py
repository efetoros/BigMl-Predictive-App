filename = " "
pickle_store =" " 

import pickle
import os
os.environ['BIGML_USERNAME'] = "efetoros"
os.environ['BIGML_API_KEY'] = "471ae5485d74ceeb2e911e0c1d37edda58cf79d3"
import bigml
from bigml.ensemble import Ensemble
from bigml.model import Model
from bigml.api import BigML
from tkinter import *
API = BigML()


LIBRARY = API.create_library("""
			
(define (sample-dataset ds-id rate oob)
		(create-dataset ds-id {"sample_rate" rate
								"out_of_bag" oob
								"seed" "whizzml-example"}))

(define (split-dataset ds-id rate)
		(list (sample-dataset ds-id rate false)
				(sample-dataset ds-id rate true)))

(define (quality-measure ev-id)
	(let (ev (fetch (wait ev-id)))
	(ev ["result" "model" "average_f_measure"])))

(define (model-or-ensemble training-set-id)
	(let (ids (split-dataset training-set-id 0.8) 
      	train-id (ids 0) 
      	test-id (ids 1) 
      	m-id (create-model train-id)
	  	e-id (create-ensemble train-id {"number_of_models" 15})
	  	m-f (quality-measure (create-evaluation m-id test-id))
	  	e-f (quality-measure (create-evaluation e-id test-id)))
	(log-info "model f " m-f " / ensemble f " e-f)
	(if (> m-f e-f) m-id e-id)))


(define (feature-names dataset-id ids)
  (let (fields ((fetch dataset-id) "fields"))
    (map (lambda (id) (fields [id "name"])) ids)))

(define (default-inputs dataset-id obj-id)
  (let (fields ((fetch dataset-id) "fields")
        fids (keys fields))
    (filter (lambda (k) (and (fields [k "preferred"] false) (not (= obj-id k))))
            fids)))

(define (make-models dataset-id obj-field selected potentials)
  (let (model-req {"dataset" dataset-id "objective_field" obj-field}
        make-req (lambda (fid)
                   (assoc model-req "input_fields" (cons fid selected)))
        all-reqs (map make-req potentials))
    (create-and-wait* "model" all-reqs)))

(define (select-feature test-dataset-id potentials model-ids)
  (let (eval-req {"dataset" test-dataset-id}
        make-req (lambda (mid) (assoc eval-req "model" mid))
        all-reqs (map make-req model-ids)
        evs (map fetch (create-and-wait* "evaluation" all-reqs))
        vs (map (lambda (ev) (ev ["result" "model" "average_phi"] 0)) evs))
        (make-map potentials vs)))

(define (get-objective ds-id obj-id)
  (let (obj-id (if (empty? obj-id)
                   (dataset-get-objective-id ds-id)
                   obj-id)
        otype ((fetch ds-id) ["fields" obj-id "optype"] "missing"))
    (when (not (= "categorical" otype))
      (raise (str "The dataset's objective field must be categorical, "
                  "but is " otype)))
    obj-id))


(define (output-features dataset-id)
  (let (obj-id (get-objective dataset-id "")
        potentials (default-inputs dataset-id obj-id)
        splits (split-dataset dataset-id 0.5)
        train-id (nth splits 0)
        test-id (nth splits 1)
        model-ids (make-models dataset-id obj-id [] potentials))
  (select-feature test-id potentials model-ids)))

	""")
API.ok(LIBRARY)


source = API.create_source(filename)
dataset = API.create_dataset(source)
model = API.create_model(dataset)
source_id = source["object"]['resource']
dataset_id = dataset["object"]['resource']


#create training and testing sets

SCRIPT_training_and_testing = API.create_script("(split-dataset ds-id rate)", {
				"imports": [LIBRARY['resource']],
				"inputs": [{"name": "ds-id", "type": "string"},
				{"name": "rate", "type": "number"}]
		})
API.ok(SCRIPT_training_and_testing)

EXECUTION_training_and_testing= API.create_execution(SCRIPT_training_and_testing['resource'], {'inputs': 
	[["ds-id", dataset_id],
	["rate", .75]]})
API.ok(EXECUTION_training_and_testing)
training_and_testing = EXECUTION_training_and_testing['object']['execution']['result']
training_set = training_and_testing[0]
testing_set = training_and_testing[1]


# #Score each feature on a phi score by bigml

SCRIPT_features = API.create_script("(output-features ds-id)", {
				"imports": [LIBRARY['resource']],
				"inputs": [{"name": "ds-id", "type": "string"}]
		})
API.ok(SCRIPT_features)

EXECUTION_features= API.create_execution(SCRIPT_features['resource'], {'inputs': [["ds-id", dataset_id]]})
API.ok(EXECUTION_features)
features_average_phi_score = EXECUTION_features['object']['execution']['result']

relevant_features = [] 
for key in features_average_phi_score:
	if features_average_phi_score.get(key) > .75 and len(relevant_features) <=10:
		relevant_features.append(key)


# Based on feature-ids, retrieve the names.
SCRIPT_features_names = API.create_script("(feature-names ds-id ids)", {
				"imports": [LIBRARY['resource']],
				"inputs": [{"name": "ds-id", "type": "string"},
				{"name": "ids", "type": "list"}]
		})
API.ok(SCRIPT_features)

EXECUTION_features_names= API.create_execution(SCRIPT_features_names['resource'], {'inputs': 
	[["ds-id", dataset_id],
	["ids", relevant_features]]})
API.ok(EXECUTION_features_names)
feature_names = EXECUTION_features_names['object']['execution']['result']


#Choose a model or ensemble

SCRIPT_model_or_ensemble = API.create_script("(model-or-ensemble ts-id)", {
				"imports": [LIBRARY['resource']],
				"inputs": [{"name": "ts-id", "type": "string"}]
		})
API.ok(SCRIPT_model_or_ensemble)

EXECUTION_model_or_enemble = API.create_execution(SCRIPT_model_or_ensemble['resource'], {'inputs': [["ts-id",training_set]]})
API.ok(EXECUTION_model_or_enemble)

model_or_ensemble = EXECUTION_model_or_enemble["object"]["execution"]["result"]


#Locally store the model or ensemble

if model_or_ensemble[:1] == 'e':
	global local_ensemble
	local_ensemble = Ensemble(model_or_ensemble)
	picklEoR= local_ensemble
else:
	global local_model
	local_model = Model(model_or_ensemble)
	picklEoR= local_model

#batch prediction to check if the model is accurate
batch_prediction = API.create_batch_prediction(model_or_ensemble, testing_set,{"all_fields": True})
API.ok(batch_prediction)
API.download_batch_prediction(batch_prediction,
                              filename= (filename[:-4]+ "-Model-or-Ensemble-Check.csv"))

#Store the data the has been created from this python file
f = open(pickle_store, 'wb')
pickle.dump([feature_names,model_or_ensemble,picklEoR], f)
f.close()












