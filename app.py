import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# 页面全局设置 (UI 设计)
st.set_page_config(page_title="慧眼识糖 AI", page_icon="🩺", layout="wide")

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

st.title("🩺 “慧眼识糖”：AI 糖尿病风险预测与数字管理平台")
st.markdown("---")

# 1. 加载模型
@st.cache_resource
def load_model():
    return joblib.load('lgbm_diabetes_model.pkl')


try:
    model = load_model()

    feature_names = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack',
                     'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
                     'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']
except FileNotFoundError:
    st.error("找不到模型文件 lgbm_diabetes_model.pkl，请确保已运行保存模型的代码！")
    st.stop()
# 2. 侧边栏：收集用户输入 (全量 21 维特征，分级显示)
st.sidebar.header("📝 居民健康特征智能录入")
st.sidebar.markdown("为了获得最精准的预测，请完善以下信息：")

# 🔴 第一部分：核心健康与体征指标 (主要特征)
st.sidebar.markdown("### 🔴 核心健康与体征指标 (主要特征)")

# 1. 年龄
age_dict = {"18-24 岁": 1, "25-29 岁": 2, "30-34 岁": 3, "35-39 岁": 4, "40-44 岁": 5, "45-49 岁": 6, "50-54 岁": 7,
            "55-59 岁": 8, "60-64 岁": 9, "65-69 岁": 10, "70-74 岁": 11, "75-79 岁": 12, "80岁及以上": 13}
age = age_dict[st.sidebar.selectbox("👴👦 您的年龄段是？", list(age_dict.keys()), index=5)]

# 2. BMI
col_h, col_w = st.sidebar.columns(2)
height = col_h.number_input("身高(cm)", min_value=100.0, max_value=250.0, value=170.0, step=1.0)
weight = col_w.number_input("体重(kg)", min_value=30.0, max_value=200.0, value=70.0, step=1.0)
bmi = weight / ((height / 100) ** 2)
st.sidebar.info(f"👉 系统自动计算 BMI: **{bmi:.1f}**")

# 3. 总体健康自评
gen_hlth_dict = {"1 = 极好 (精力充沛，极少生病)": 1, "2 = 很好 (偶有微恙，恢复极快)": 2, "3 = 好 (无大病，基本正常)": 3,
                 "4 = 一般 (亚健康，容易疲劳)": 4, "5 = 差 (长期患病，躯体受限)": 5}
gen_hlth = gen_hlth_dict[st.sidebar.selectbox("🏥 总体健康自评？", list(gen_hlth_dict.keys()), index=2)]

# 4-6. 核心病史
high_bp = st.sidebar.selectbox("🩸 是否有【高血压】?", [0, 1], format_func=lambda x: "是" if x == 1 else "否")
high_chol = st.sidebar.selectbox("🫀 是否有【高胆固醇】?", [0, 1], format_func=lambda x: "是" if x == 1 else "否")
heart_disease = st.sidebar.selectbox("💔 是否有过【冠心病或心肌梗塞】?", [0, 1],
                                     format_func=lambda x: "是" if x == 1 else "否")
stroke = st.sidebar.selectbox("🧠 是否有过【中风】史?", [0, 1], format_func=lambda x: "是" if x == 1 else "否")
diff_walk = st.sidebar.selectbox("🚶‍♂️ 是否有【步行或爬楼梯困难】?", [0, 1],
                                 format_func=lambda x: "是" if x == 1 else "否")

# 7-8. 近期身心状态
ment_hlth = st.sidebar.slider("😔 过去30天【心理压力大/情绪低落】天数", 0, 30, 0)
phys_hlth = st.sidebar.slider("🤒 过去30天【身体不适/生病】天数", 0, 30, 0)

# 🟢 第二部分：生活方式与社会经济指标 (次要特征)
st.sidebar.divider()  # 分界线
st.sidebar.markdown("### 🟢 生活习惯与社会经济 (次要特征)")

# 采用折叠面板，让界面更清爽，别人一看就知道这是辅助项
with st.sidebar.expander("点击展开填写辅助指标 (有利于提升精度)", expanded=False):
    sex = st.selectbox("🚻 性别", [0, 1], format_func=lambda x: "男性" if x == 1 else "女性", index=1)
    smoker = st.selectbox("🚬 是否抽烟 (一生中抽过100支以上)?", [0, 1], format_func=lambda x: "是" if x == 1 else "否")
    hvy_alcohol = st.selectbox("🍺 是否重度饮酒?", [0, 1], format_func=lambda x: "是" if x == 1 else "否")
    phys_act = st.selectbox("🏃‍♂️ 过去一个月是否有规律运动?", [0, 1], format_func=lambda x: "是" if x == 1 else "否",
                            index=1)
    fruits = st.selectbox("🍎 每天是否吃水果 (1次或以上)?", [0, 1], format_func=lambda x: "是" if x == 1 else "否",
                          index=1)
    veggies = st.selectbox("🥬 每天是否吃蔬菜 (1次或以上)?", [0, 1], format_func=lambda x: "是" if x == 1 else "否",
                           index=1)
    chol_check = st.selectbox("🩺 过去5年内查过胆固醇吗?", [0, 1], format_func=lambda x: "是" if x == 1 else "否",
                              index=1)
    any_healthcare = st.selectbox("🏥 是否有任何形式的医疗保险?", [0, 1], format_func=lambda x: "是" if x == 1 else "否",
                                  index=1)
    no_doc_cost = st.selectbox("💰 过去一年是否因费用问题放弃就医?", [0, 1],
                               format_func=lambda x: "是" if x == 1 else "否")

    edu_dict = {"未上过学或幼儿园": 1, "小学至初中": 2, "高中未毕业": 3, "高中毕业": 4, "大学肄业": 5, "大学毕业": 6}
    education = edu_dict[st.selectbox("🎓 最高学历", list(edu_dict.keys()), index=5)]

    income_dict = {"< $10,000": 1, "$10,000 - $15,000": 2, "$15,000 - $20,000": 3, "$20,000 - $25,000": 4,
                   "$25,000 - $35,000": 5, "$35,000 - $50,000": 6, "$50,000 - $75,000": 7, "> $75,000": 8}
    income = income_dict[st.selectbox("💵 家庭年收入等级", list(income_dict.keys()), index=7)]

# 组装完整的 21 维输入向量
user_data = {
    'HighBP': high_bp, 'HighChol': high_chol, 'CholCheck': chol_check, 'BMI': bmi, 'Smoker': smoker,
    'Stroke': stroke, 'HeartDiseaseorAttack': heart_disease, 'PhysActivity': phys_act, 'Fruits': fruits,
    'Veggies': veggies, 'HvyAlcoholConsump': hvy_alcohol, 'AnyHealthcare': any_healthcare, 'NoDocbcCost': no_doc_cost,
    'GenHlth': gen_hlth, 'MentHlth': ment_hlth, 'PhysHlth': phys_hlth, 'DiffWalk': diff_walk,
    'Sex': sex, 'Age': age, 'Education': education, 'Income': income
}
input_df = pd.DataFrame([user_data], columns=feature_names)

# 3. 核心功能区 (主界面)
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("💡 阶段一：极速智能筛查")
    if st.button("🚀 一键生成数字孪生体检报告", use_container_width=True):
        # 预测概率
        prob = model.predict_proba(input_df)[0]
        pred_class = np.argmax(prob)

        status_dict = {0: "健康 (低风险)", 1: "糖尿病前期 (极需干预)", 2: "确诊糖尿病 (高危)"}
        color_dict = {0: "green", 1: "orange", 2: "red"}

        st.markdown(f"### 当前预测状态：<span style='color:{color_dict[pred_class]}'>{status_dict[pred_class]}</span>",
                    unsafe_allow_html=True)
        st.progress(float(prob[pred_class]))
        st.write(f"**患病综合预警概率：{prob[pred_class] * 100:.1f}%**")

        # 触发 Apriori 干预罗盘
        st.markdown("### 💊 阶段二：智能靶向健康提醒")
        if high_bp == 1 and bmi >= 30:
            st.error(
                "⚠️ **系统严重警告**：监测到【高血压 + 重度肥胖】高危并发组合！\n\n数据表明该组合将使糖尿病风险激增 2.47 倍！\n\n**联合干预处方**：请立刻启动 DASH 饮食法（低钠高钾），将每日盐摄入控制在 5g 以下；每周严格保持 150 分钟中等强度有氧运动以强制降低 BMI。")
        elif ment_hlth >= 5:
            st.warning(
                "⚠️ **系统次级提醒**：您的心理承压已达亚健康临界值（过去一月>5天情绪不佳）。心理亚健康正隐性推高您的代谢风险，建议规律作息，必要时引入心理疏导干预。")
        else:
            st.success("✅ 目前数据表现良好，未触发高危多重并发症预警规则。请继续保持现在的健康生活方式与规律饮食！")

with col2:
    st.subheader("🔍 阶段三：个体危险因子白盒化溯源")
    if 'pred_class' in locals():
        st.markdown("**【核心算法引擎驱动中】**：基于博弈论 SHAP 模型，为您精准拆解推高或降低患病风险的关键特征...")

        # 计算单样本 SHAP
        explainer = shap.TreeExplainer(model)
        shap_values_raw = explainer.shap_values(input_df)

        TARGET_CLASS = 2
        if isinstance(shap_values_raw, list):
            target_shap = shap_values_raw[TARGET_CLASS][0]
        else:
            target_shap = shap_values_raw[0, :, TARGET_CLASS]

        # 整理画图数据
        exp_df = pd.DataFrame({'Feature': feature_names, 'SHAP Value': target_shap})
        exp_df['Abs Value'] = exp_df['SHAP Value'].abs()
        exp_df = exp_df.sort_values(by='Abs Value', ascending=False).head(10)  # 展现前 10 大影响因素

        # 绘制极其美观的红蓝条形图
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#ff4d4d' if val > 0 else '#4d94ff' for val in exp_df['SHAP Value']]
        ax.barh(exp_df['Feature'], exp_df['SHAP Value'], color=colors, edgecolor='black')
        ax.set_xlabel('对患病风险的额外推力 (红色为恶化，蓝色为保护)', fontsize=11)
        ax.set_title('千人千面：您的专属发病风险溯源图', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        st.pyplot(fig)
        st.markdown(
            "*(👉 阅读指南：红色条块代表该指标正在“催化”您的患病风险；蓝色条块代表该指标正在“保护”您远离疾病。条块越长，影响力越大。)")