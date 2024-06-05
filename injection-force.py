import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import math

# 定义注射力计算函数，这里仅作示意，具体公式请根据研究论文调整
def calculate_injection_force(V, v, n, mu, L_mm, A_mm2, R_mm, mu_oil, r_b_mm, l_stopper_mm, d_oil_mm):
    K1 = (8 * L_mm * A_mm2) / (math.pi * R_mm**4)
    K2 = (2 * math.pi * mu_oil * r_b_mm * l_stopper_mm) / d_oil_mm
    gamma_eff = 2 * v / (R_mm * (2*n + 1) / (3*n + 1))  # 有效剪切速率计算
    mu_eff = K1 * gamma_eff**(n - 1)  # 有效粘度计算
    f_friction = K2 * V
    F_injection = mu_eff * mu + f_friction
    return F_injection

def plot_injection_force(mu_values, force_values, title="注射力与粘度关系"):
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, force_values, 'bo-', label='实际测量值')
    plt.title(title)
    plt.xlabel('fluid viscosity (cP)')
    plt.ylabel('Injection force (N)')
    plt.legend()
    st.pyplot(plt)
    plt.close()

def exponential_model(x, a, b):
    return a * np.exp(b * x)

# Streamlit App 开始
st.title("注射力模拟与粘度依赖性评估")

# 文献引用
#image = Image.open('solubility-logo.jpg')
#st.image(image, use_column_width=True)
st.markdown("### 参考文献")
st.write("[Wu, Linke et al. [Advancing injection force modeling and viscosity-dependent injectability evaluation for prefilled syringes.](https://doi.org/10.1016/j.ejpb.2024.114221) ***Eur J Pharm Biopharm.*** 2024;197:114221.")

# 用户输入
st.header("输入以下参数以计算注射力:")
V = st.number_input("The injection rate (m/s)", min_value=0.0, step=0.001, value=0.1, format="%.3f")
v = st.number_input("Average fluid velocity (m/s)", min_value=0.0, step=0.001, value=0.1, format="%.3f")
n = st.number_input("Power law index;dimensionless", min_value=0.0, step=0.001, value=0.1, format="%.3f")
mu = st.number_input("Flow consistency index (pa s^n)", min_value=0.0, step=0.001, value=0.1, format="%.3f")
L_mm = st.number_input("Syringe tip length (mm)", min_value=0.0, step=0.001, value=1.0, format="%.3f")
A_mm2 = st.number_input("Cross sectional area of plunger stopper  (mm^2)", min_value=0.0, step=0.001, value=1.0, format="%.3f")
R_mm = st.number_input("Needle inner radius (mm)", min_value=0.0, step=0.001, value=1.5, format="%.3f")
mu_oil = st.number_input("Silicone oil viscosity (pa s^n)", min_value=0.0, step=0.001, value=0.5, format="%.3f")
r_b_mm = st.number_input("The syringe barrel radius (mm)", min_value=0.0, step=0.001, value=2.0, format="%.3f")
l_stopper_mm = st.number_input("The length of the stopper in contact with glass (mm)", min_value=0.0, step=0.001, value=5.0, format="%.3f")
d_oil_mm = st.number_input("The thickness of silicone oil (mm)", min_value=0.0, step=0.001, value=0.1, format="%.3f")

# 计算并显示注射力
st.header("计算结果")
force = calculate_injection_force(V,v,n, mu, L_mm/1000, A_mm2/1000000, R_mm/1000, mu_oil, r_b_mm/1000, l_stopper_mm/1000, d_oil_mm/1000)  # 注意单位转换
st.write(f"预测注射力为: {force:.2f} N")

# 数据模拟与绘图
# 设置英文字体
plt.rcParams['font.family'] = 'Arial'  # 或者 'Helvetica' 或其他英文字体
plt.rcParams['font.size'] = 14

# 确保负号正常显示
plt.rcParams['axes.unicode_minus'] = False

# 数据模拟与绘图
if st.checkbox("模拟不同粘度下的注射力"):
    st.header("Injection Force vs. Viscosity Simulation")
    
    # 用户选择粘度范围的开始和结束值
    mu_start = st.slider("Viscosity Range Start (cP)", min_value=0.1, max_value=50.0, value=0.1, step=0.1)
    mu_end = st.slider("Viscosity Range End (cP)", min_value=0.1, max_value=50.0, value=50.0, step=0.1)
    
    # 用户选择mu_range中的点数
    num_points = st.slider("Number of Viscosity Points", min_value=10, max_value=100, value=10, step=1)
    
    # 生成粘度范围
    mu_range = np.linspace(mu_start, mu_end, num_points)
    
    # 计算注射力
    force_range = [calculate_injection_force(V, mu_i, L_mm/1000, A_mm2/1000000, R_mm/1000, mu_oil, r_b_mm/1000, l_stopper_mm/1000, d_oil_mm/1000) for mu_i in mu_range]
    
    # 拟合数据以展示可能的非线性关系
    popt, _ = curve_fit(exponential_model, mu_range, force_range)
    st.write(f"Fitted Curve Parameters: a = {popt[0]:.2f}, b = {popt[1]:.2f}")
    
    # 绘制原始数据点
    plot_injection_force(mu_range, force_range, title="Injection Force vs. Viscosity")
    
    # 添加复选框来决定是否显示拟合曲线
    if st.checkbox("Show Fitted Curve"):
        # 绘制拟合曲线
        plot_injection_force(mu_range, exponential_model(mu_range, *popt), title="Fitted Curve vs. Actual Measurements")

st.sidebar.header("关于此应用")
st.sidebar.info("此应用基于Wu等人2024年的研究，旨在帮助用户理解并预测预充式注射器中注射力与药物粘度之间的关系。输入相关参数，即可获得模拟的注射力数值及可视化结果。")

# 结束
st.write("希望此应用能为您的研究或产品设计带来便利！")
