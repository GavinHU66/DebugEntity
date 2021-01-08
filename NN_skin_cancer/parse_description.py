import json
import pandas as pd
import glob

df = pd.DataFrame(columns=['name', 'age_approx', 'anatomy', 'malign',
 'diagnosis', 'diagnosis_confirm_type', 'melanocytic', 'sex', 'image_type', 'size_x', 'size_y'])
file_names = glob.glob(r".\tools\ISIC-Archive-Downloader\Data\Descriptions\*")
for name in file_names:
    print(name)
    with open(name, 'r') as f:
        description = json.load(f)
        sex = description['meta']['clinical'].get('sex')
        melanocytic = description['meta']['clinical'].get('melanocytic')
        df = df.append({
            'name': name.split('\\')[-1], 
            'age_approx': description['meta']['clinical'].get('age_approx'),
            'anatomy': description['meta']['clinical'].get('anatom_site_general'),
            'malign': description['meta']['clinical'].get('benign_malignant'),
            'diagnosis_confirm_type': description['meta']['clinical'].get('diagnosis_confirm_type'),
            'melanocytic': melanocytic,
            'sex': sex,
            'image_type': description['meta']['acquisition'].get('image_type'),
            'size_x': description['meta']['acquisition'].get('pixelsX'),
            'size_y': description['meta']['acquisition'].get('pixelsY')
        }, ignore_index=True)

df = pd.get_dummies(df, columns=['anatomy', 'malign' , 'diagnosis', 'diagnosis_confirm_type', 'melanocytic', 'sex', 'image_type'])
df.to_csv('./description.csv')
print(df.info())
