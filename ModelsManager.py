import json
import hashlib
from datetime import datetime
import git
import pandas as pd

def multi_dic(dic, prefix=""):
    arr = []
    for key, value in dic.items():
        name = prefix + key
        if type(value) == type({}):
            content = multi_dic(value, name + "_")
            for c in content:
                arr.append(c)
        else:
            arr.append([name, value])
    return arr

class ModelsManager():
    def __init__(self, savefile="model_cache/Models.csv"):
        self.models = {}
        self.savefile = savefile
        self.df = None
        self.df_exist = False
        self.best_score = (194 / 2)
        # print("Model created")

    def get_model_key(self, params):
        json_string = json.dumps(params, indent=4,
                  default=lambda o: '<not serializable>')
        key = hashlib.md5(json_string.encode())
        key = key.hexdigest()
        # print("Key is: ", key)
        return key

    def new_model(self, params):
        model = {}
        model["params"] = params
        model["scores"] = {}
        model["meta"] = {}
        model["meta"]["datetime"] = datetime.now().strftime("%d-%m %H:%M")
        repo = git.Repo(search_parent_directories=True)
        model["meta"]["git"] = repo.head.object.hexsha
        return model

    def new_score(self, params, score, score_name, last=False):
        key = self.get_model_key(params)
        if not key in self.models:
            self.models[key] = self.new_model(params)
        self.models[key]["scores"][score_name] = score
        # print("New score added: ", score)
        if last:
            self.add_line(key)
        if score['weird_metric'] > self.best_score:
            self.best_score = score['weird_metric']
            return True
        return False

    def add_line(self, key):
        df_line = multi_dic(self.models[key])
        df_line.sort(key=lambda c: c[0])
        # print(df_line)
        
        if not self.df_exist:
            name = []
            valu = [[]]
            for n, v in df_line:
                name.append(n)
                valu[0].append(v)
            self.df = pd.DataFrame(valu, columns=name)
            self.df_exist = True
        else:
            line = {name: value for name, value in df_line}
            self.df = self.df.append(line, ignore_index=True)
        self.df.to_csv(self.savefile, index=False)
        # print("File saved !")
