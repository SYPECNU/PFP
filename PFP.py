import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

# 载入数据
@st.cache_data
def load_data():
    data1 = pd.read_csv("ML有用肌力_imputed.csv", encoding='gbk')  # 请确保加速.csv已经上传
    data1.dropna(inplace=True)
    data1.columns = ['Score', 'Right erector spinae', 'Left erector spinae', 'Right external oblique', 
                     'Left external oblique', 'Gluteus maximus', 'Gluteus medius', 'Gluteus minimus', 
                     'Semimembranous', 'Semitendinosus', 'Biceps femoris', 'Adductor brevis', 'Adductor magnus', 
                     'Tensor fascia lata', 'Rectus femoris', 'Medial gastrocnemius', 'Lateral gastrocnemius', 
                     'Soleus', 'Flexor digitorum longus', 'Flexor pollicis longus']
    return data1

data1 = load_data()

# 提取特征和标签
X = data1[['Right erector spinae', 'Left erector spinae', 'Right external oblique', 'Left external oblique', 
           'Gluteus maximus', 'Gluteus medius', 'Gluteus minimus', 'Semimembranous', 'Semitendinosus', 
           'Biceps femoris', 'Adductor brevis', 'Adductor magnus', 'Tensor fascia lata', 'Rectus femoris', 
           'Medial gastrocnemius', 'Lateral gastrocnemius', 'Soleus', 'Flexor digitorum longus', 'Flexor pollicis longus']]

y = data1[['Score']]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练XGBoost模型
model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=1, min_child_weight=9,
                         learning_rate=0.3, n_estimators=200, subsample=0.7, max_depth=6)
model.fit(X_train, y_train)

# 创建Streamlit界面
st.title("Patellofemoral Pain Prediction")

# 显示应用说明
st.write("""
This application is designed to predict and identify personalized risk factors related to increased patellofemoral pain.
After entering your muscle activation parameters on the right, the model will predict your patellofemoral pain intensity 
based on the muscle activation patterns. If the predicted pain level is high, you can actively adjust your lower limb muscle 
strength to reduce patellofemoral stress and prevent worsening pain.
""")

# 在侧边栏中添加用户输入
st.sidebar.header("Input Parameters")

# 各种特征的输入控件（修改特征值范围为0.00到100.00）
right_erector_spinae = st.sidebar.slider("Right erector spinae", min_value=0.00, max_value=100.00, value=50.00)
left_erector_spinae = st.sidebar.slider("Left erector spinae", min_value=0.00, max_value=100.00, value=50.00)
right_external_oblique = st.sidebar.slider("Right external oblique", min_value=0.00, max_value=100.00, value=50.00)
left_external_oblique = st.sidebar.slider("Left external oblique", min_value=0.00, max_value=100.00, value=50.00)
gluteus_maximus = st.sidebar.slider("Gluteus maximus", min_value=0.00, max_value=100.00, value=50.00)
gluteus_medius = st.sidebar.slider("Gluteus medius", min_value=0.00, max_value=100.00, value=50.00)
gluteus_minimus = st.sidebar.slider("Gluteus minimus", min_value=0.00, max_value=100.00, value=50.00)
semimembranous = st.sidebar.slider("Semimembranous", min_value=0.00, max_value=100.00, value=50.00)
semitendinosus = st.sidebar.slider("Semitendinosus", min_value=0.00, max_value=100.00, value=50.00)
biceps_femoris = st.sidebar.slider("Biceps femoris", min_value=0.00, max_value=100.00, value=50.00)
adductor_brevis = st.sidebar.slider("Adductor brevis", min_value=0.00, max_value=100.00, value=50.00)
adductor_magnus = st.sidebar.slider("Adductor magnus", min_value=0.00, max_value=100.00, value=50.00)
tensor_fascia_lata = st.sidebar.slider("Tensor fascia lata", min_value=0.00, max_value=100.00, value=50.00)
rectus_femoris = st.sidebar.slider("Rectus femoris", min_value=0.00, max_value=100.00, value=50.00)
medial_gastrocnemius = st.sidebar.slider("Medial gastrocnemius", min_value=0.00, max_value=100.00, value=50.00)
lateral_gastrocnemius = st.sidebar.slider("Lateral gastrocnemius", min_value=0.00, max_value=100.00, value=50.00)
soleus = st.sidebar.slider("Soleus", min_value=0.00, max_value=100.00, value=50.00)
flexor_digitorum_longus = st.sidebar.slider("Flexor digitorum longus", min_value=0.00, max_value=100.00, value=50.00)
flexor_pollicis_longus = st.sidebar.slider("Flexor pollicis longus", min_value=0.00, max_value=100.00, value=50.00)

# 用户输入的特征值
user_input = np.array([[right_erector_spinae, left_erector_spinae, right_external_oblique, left_external_oblique, 
                        gluteus_maximus, gluteus_medius, gluteus_minimus, semimembranous, semitendinosus, 
                        biceps_femoris, adductor_brevis, adductor_magnus, tensor_fascia_lata, rectus_femoris, 
                        medial_gastrocnemius, lateral_gastrocnemius, soleus, flexor_digitorum_longus, flexor_pollicis_longus]])

# 预测
predicted_pain_score = model.predict(user_input)

# 显示预测结果
st.write(f"Predicted Patellofemoral Pain Intensity: {predicted_pain_score[0]:.2f}")
