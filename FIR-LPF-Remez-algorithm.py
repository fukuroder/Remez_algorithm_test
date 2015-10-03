# -*- coding: utf-8 -*-
import numpy, math

# 理想のローパスフィルタ
def ideal_lowpass_filter(x, w0):
    return 1.0 if x < w0 else 0.0

# 三角多項式
def tri_polynomial(x, list_a):
    return sum(a*math.cos(x*k) for k,a in enumerate(list_a))

# 三角多項式の一次導関数
def d_tri_polynomial(x, list_a):
    return sum(-a*k*math.sin(x*k) for k,a in enumerate(list_a))

# 三角多項式の二次導関数
def dd_tri_polynomial(x, list_a):
    return sum(-a*k*k*math.cos(x*k) for k,a in enumerate(list_a))

# 極値点を全て求める
def search_extreme_points(
    list_a,      # 三角多項式の係数列
    div):        # 分割数

    # f(x)
    def f(x):
        return d_tri_polynomial(x, list_a);

    # f(x)の導関数
    def df(x):
        return dd_tri_polynomial(x, list_a);

    # ニュートン法
    def newton(x0):
        x = x0
        for _ in range(100):
            x_next = x - f(x)/df(x)
            if abs(x_next - x) < 1.0e-8:
                break
            x = x_next
        else:
            print("[WARNING]Newton method iterations reached the limit(x=" + str(x0) + ")")
        return x

    # 点列を生成
    check_points = numpy.linspace(0.0, math.pi, div)

    # f(x)の符号が変わる区間を探す
    sign_reverse_section = \
        [p for p in zip(check_points, check_points[1:]) if f(p[0])*f(p[1]) <= 0.0]

    # f(x)=0になるxを全て求める
    return [newton(x) for x,_ in sign_reverse_section]

###########################
# Remez algorithm --step1--
# 検査点の初期値を生成
###########################
def initialize_extreme_points(
    n,  # 三角多項式の次数
    w0, # 遮断周波数
    h): # 遷移幅

    # 通過域点数＋阻止域点数
    num_point = n+2;

    # 通過域点数
    num_passband_point = int(num_point*w0/math.pi)

    # 阻止域点数
    num_stopband_point = num_point-num_passband_point

    # 初期点を生成
    return numpy.append(
        numpy.linspace(     0.0, w0-0.5*h, num_passband_point),
        numpy.linspace(w0+0.5*h,  math.pi, num_stopband_point))

##########################################################
# Remez algorithm --step2--
# 検査点(x[0],...,x[n+1])での誤差の符号が交互に反転するような
# 三角多項式係数(a[0],...a[n])と誤差dを求める
##########################################################
def update_tri_polynomial_coefficients(
    list_x, # 検査点
    w0):    # 遮断周波数

    # 行列A作成
    matrix_A = numpy.array(
        [[math.cos(x*k) for k in range(len(list_x)-1)] + [(-1)**j] \
                        for j,x in enumerate(list_x)])

    # ベクトルb作成
    vector_b = numpy.array([ideal_lowpass_filter(x, w0) for x in list_x])

    # 連立方程式を解く
    u = numpy.linalg.solve(matrix_A, vector_b)

    # a[0],...,a[n], d
    return u[:-1], u[-1]

###########################################
# Remez algorithm --step3--
# 誤差関数の絶対値が最大となるn+2個の点を求める
###########################################
def update_maximum_error_points(
    list_a, # 三角多項式の係数列
    w0,     # 遮断周波数
    h):     # 遷移幅

    # 三角多項式の次数
    n = len(list_a)-1

    # 三角多項式の極値点を求める
    extreme_points = search_extreme_points(list_a, (n+2)*10)

    # 遷移域の開始終了位置も極値点だと思う
    extreme_points.append(w0-h*0.5)
    extreme_points.append(w0+h*0.5)
    extreme_points.sort()

    if len(extreme_points) == n+1:
        # 極値点数がn+1ならば端点(x=pi)も極値点だと思う
        extreme_points.append(math.pi)
        return extreme_points

    elif len(extreme_points) == n+2:
        # そのままでOK
        return extreme_points

    elif len(extreme_points) == n+3:
        # 極値点数がn+3ならば端点(x=0)を極値点から外す
        extreme_points.pop(0)
        return extreme_points

    else:
        raise Exception("[ERROR]number of extreme point " + \
            str(n+2) + "->" + str(len(extreme_points)))

#############################################################
# Remez algorithm --step4--
# 収束判定（誤差の絶対値が等しくかつ符号が交互に並んでいれば終了）
#############################################################
def check_convergence(
    list_a, # 三角多項式の係数列
    list_x, # 検査点列
    w0):    # 遮断周波数

    # 誤差関数
    def ef(x):
        return tri_polynomial(x, list_a)-ideal_lowpass_filter(x, w0)

    return numpy.var([ef(x)*(-1)**k for k,x in enumerate(list_x)]) < 1.0e-12

######################
# Remez algorithm 本体
######################
def remez(
    order,         # フィルタの次数
    w0,            # 遮断周波数
    h,             # 遷移幅
    max_iter=100): # 最大反復回数

    n = (order-1)//2 # 三角多項式の次数

    # Remez algorithm --step1--
    # 検査点の初期値を生成
    list_x = initialize_extreme_points(n, w0, h)

    for count in range(1, max_iter+1):
        # Remez algorithm --step2--
        # 検査点(x[0],...,x[n+1])での誤差の符号が交互に反転するような
        # 三角多項式係数(a[0],...a[n])と誤差dを求める
        list_a, d = update_tri_polynomial_coefficients(list_x, w0)

        # Remez algorithm --step3--
        # 誤差関数の絶対値が最大となるn+2個の点を求める
        list_x = update_maximum_error_points(list_a, w0, h)

        # Remez algorithm --step4--
        # 収束判定（誤差の絶対値が等しくかつ符号が交互に並んでいれば終了）
        if check_convergence(list_a, list_x, w0):
            # cos関数系からexp関数系の表現に変換
            list_h = [a*0.5 for a in reversed(list_a[1:])] + \
                     [list_a[0]] + \
                     [a*0.5 for a in list_a[1:]]
            return list_h, d, list_x, count
    else:
        raise Exception("[ERROR]Remez algorithm failed")

if __name__ == '__main__':
    print("filter order(odd number):",end="")
    order = int(input())
    if order<3 or order%2 == 0:
        raise Exception("[ERROR]Please input an odd number.")

    print("""
=======================
[ideal low-pass filter]

1+--------+
          |
0+        +--------+
 0        w0       pi
=======================
cutoff frequency(w0):""", end="")
    w0 = float(input())
    if w0 <= 0.0 or math.pi <= w0:
        raise Exception("[ERROR]Please input range (0.0, pi).")

    print("""
=======================
[transition width]

1+-----****
       <--h-->
0+        ****-----+
 0        w0       pi
=======================
transition width(h):""", end="")
    h = float(input())
    if w0-h*0.5 <= 0.0 or math.pi <= w0+h*0.5:
        raise Exception("[ERROR]Transition width is too large.")

    print("calculating...")
    list_h, d, list_x, count = remez(order, w0, h)

    print()
    for h in list_h:
        print(h)
