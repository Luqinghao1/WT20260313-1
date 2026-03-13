/*
 * 文件名: modelsolver01-06.cpp
 * 文件作用: 压裂水平井复合模型 Group 1 (Model 1-36) 核心计算实现
 * 功能描述:
 * 1. 实现了基于 Stehfest 数值反演算法的 Laplace 变换求解。
 * 2. 实现了边界元方法 (BEM) 求解多段压裂水平井的压力响应。
 * 3. [新增] 实现了 Fair 模型和 Hegeman 模型的变井储效应叠加计算。
 * 4. 严格按照 modelsolver1.csv 定义的顺序处理 36 种模型组合。
 */

#include "modelsolver01-06.h"
#include "pressurederivativecalculator.h"

#include <Eigen/Dense>
#include <boost/math/special_functions/bessel.hpp>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <QDebug>
#include <QtConcurrent>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 二维点结构体
struct Point2D { double x; double y; };

// 安全的 Bessel K 函数 (防止溢出)
static double safe_bessel_k(int v, double x) {
    if (x < 1e-15) x = 1e-15;
    try { return boost::math::cyl_bessel_k(v, x); } catch (...) { return 0.0; }
}

// 安全的 Bessel I 标度函数 (防止溢出，大参数使用渐近近似)
static double safe_bessel_i_scaled(int v, double x) {
    if (x < 0) x = -x;
    if (x > 600.0) return 1.0 / std::sqrt(2.0 * M_PI * x);
    try { return boost::math::cyl_bessel_i(v, x) * std::exp(-x); } catch (...) { return 0.0; }
}

ModelSolver01_06::ModelSolver01_06(ModelType type)
    : m_type(type), m_highPrecision(true), m_currentN(0) {
    // 默认初始化 Stehfest 系数 N=10
    precomputeStehfestCoeffs(10);
}

ModelSolver01_06::~ModelSolver01_06() {}

void ModelSolver01_06::setHighPrecision(bool high) { m_highPrecision = high; }

// 获取模型名称描述
QString ModelSolver01_06::getModelName(ModelType type, bool verbose)
{
    int id = (int)type + 1; // 转换为 1-36 的 ID
    QString baseName;
    QString subType;

    // 确定储层类型 (每12个一组)
    if (id <= 12) {
        baseName = QString("夹层型储层试井解释模型%1").arg(id);
        subType = "夹层型+夹层型";
    } else if (id <= 24) {
        baseName = QString("夹层型储层试井解释模型%1").arg(id);
        subType = "夹层型+均质";
    } else {
        baseName = QString("径向复合模型%1").arg(id - 24); // 径向复合从1开始编号
        subType = "均质+均质";
    }

    if (!verbose) return baseName;

    // 解析井储模型 (4种循环: 定井储, 线源解, Fair, Hegeman)
    int rem4 = (id - 1) % 4;
    QString strStorage;
    if (rem4 == 0) strStorage = "定井储";
    else if (rem4 == 1) strStorage = "线源解";
    else if (rem4 == 2) strStorage = "Fair模型";
    else strStorage = "Hegeman模型";

    // 解析边界 (12个一组中的 3种大类)
    // 0-3: 无限大, 4-7: 封闭, 8-11: 定压
    int groupIdx = (id - 1) % 12;
    QString strBoundary;
    if (groupIdx < 4) strBoundary = "无限大外边界";
    else if (groupIdx < 8) strBoundary = "封闭边界";
    else strBoundary = "定压边界";

    return QString("%1\n(%2、%3、%4)").arg(baseName).arg(strStorage).arg(strBoundary).arg(subType);
}

QVector<double> ModelSolver01_06::generateLogTimeSteps(int count, double startExp, double endExp)
{
    QVector<double> t;
    if (count <= 0) return t;
    t.reserve(count);
    for (int i = 0; i < count; ++i) {
        double exponent = startExp + (endExp - startExp) * i / (count - 1);
        t.append(pow(10.0, exponent));
    }
    return t;
}

// 计算理论曲线主函数
ModelCurveData ModelSolver01_06::calculateTheoreticalCurve(const QMap<QString, double>& params, const QVector<double>& providedTime)
{
    QVector<double> tPoints = providedTime;
    if (tPoints.isEmpty()) tPoints = generateLogTimeSteps(100, -3.0, 3.0);

    // --- 1. 读取物理参数 ---
    double phi = params.value("phi", 0.05);
    double mu = params.value("mu", 0.5);
    double B = params.value("B", 1.05);
    double Ct = params.value("Ct", 5e-4);
    double q = params.value("q", 5.0);
    double h = params.value("h", 20.0);
    double kf = params.value("kf", 1e-3);
    double L = params.value("L", 1000.0);

    // [新增] 变井储参数
    double alpha = params.value("alpha", 0.1);
    double C_phi = params.value("C_phi", 1e-4);

    if (L < 1e-9) L = 1000.0;
    // 参数有效性检查
    if (phi < 1e-12 || mu < 1e-12 || Ct < 1e-12 || kf < 1e-12) {
        return std::make_tuple(tPoints, QVector<double>(tPoints.size(), 0.0), QVector<double>(tPoints.size(), 0.0));
    }

    // --- 2. 转换无因次时间 ---
    double td_coeff = 14.4 * kf / (phi * mu * Ct * pow(L, 2));
    QVector<double> tD_vec;
    tD_vec.reserve(tPoints.size());
    for(double t : tPoints) tD_vec.append(td_coeff * t);

    // --- 3. 设置 Stehfest 参数 ---
    QMap<QString, double> calcParams = params;
    int N = (int)calcParams.value("N", 10);
    if (N < 4 || N > 18 || N % 2 != 0) N = 10;
    calcParams["N"] = N;
    precomputeStehfestCoeffs(N);

    if (!calcParams.contains("nf") || calcParams["nf"] < 1) calcParams["nf"] = 1;
    if (!calcParams.contains("n_seg")) calcParams["n_seg"] = 6;

    // --- 4. 计算 Laplace 空间解并反演 ---
    QVector<double> PD_vec, Deriv_vec;
    // 绑定 flaplace_composite 函数
    auto func = std::bind(&ModelSolver01_06::flaplace_composite, this, std::placeholders::_1, std::placeholders::_2);
    // 这里只计算基础压力 PD，导数暂不计算(因为后续要叠加 Fair/Hegeman)
    calculatePDandDeriv(tD_vec, calcParams, func, PD_vec, Deriv_vec);

    // --- 5. 转换为有因次压力并叠加变井储效应 ---
    double p_coeff = 1.842e-3 * q * mu * B / (kf * h);
    QVector<double> finalP(tPoints.size()), finalDP(tPoints.size());

    // 判断井储类型: 0:Const, 1:Line, 2:Fair, 3:Hegeman
    int storageType = (int)m_type % 4;

    for(int i=0; i<tPoints.size(); ++i) {
        // 基础压力 (MPa)
        double p_base = p_coeff * PD_vec[i];

        // [新增] Fair / Hegeman 模型叠加
        // 注意: 叠加项是在有因次压力 (MPa) 上进行的
        if (storageType == 2) { // Fair 模型
            if (std::abs(alpha) > 1e-9)
                p_base += C_phi * (1.0 - std::exp(-tPoints[i] / alpha));
        } else if (storageType == 3) { // Hegeman 模型
            if (std::abs(alpha) > 1e-9)
                p_base += C_phi * std::erf(tPoints[i] / alpha);
        }

        finalP[i] = p_base;
    }

    // --- 6. 重新计算导数 ---
    // 由于叠加了变井储项，压力曲线形态改变，需要对最终压力曲线进行数值求导
    if (tPoints.size() > 2) {
        finalDP = PressureDerivativeCalculator::calculateBourdetDerivative(tPoints, finalP, 0.1);
    } else {
        finalDP.fill(0.0);
    }

    return std::make_tuple(tPoints, finalP, finalDP);
}

// Stehfest 反演实现
void ModelSolver01_06::calculatePDandDeriv(const QVector<double>& tD, const QMap<QString, double>& params,
                                           std::function<double(double, const QMap<QString, double>&)> laplaceFunc,
                                           QVector<double>& outPD, QVector<double>& outDeriv)
{
    int numPoints = tD.size();
    outPD.resize(numPoints);
    outDeriv.resize(numPoints);

    int N = (int)params.value("N", 10);
    double ln2 = 0.6931471805599453;
    double gamaD = params.value("gamaD", 0.0); // 渗透率模量(压敏)

    QVector<int> indexes(numPoints);
    std::iota(indexes.begin(), indexes.end(), 0);

    auto calculateSinglePoint = [&](int k) {
        double t = tD[k];
        if (t <= 1e-10) { outPD[k] = 0.0; return; }

        double pd_val = 0.0;
        for (int m = 1; m <= N; ++m) {
            double z = m * ln2 / t;
            double pf = laplaceFunc(z, params); // 调用 Laplace 解
            if (std::isnan(pf) || std::isinf(pf)) pf = 0.0;
            pd_val += getStehfestCoeff(m, N) * pf;
        }

        double pd_real = pd_val * ln2 / t;
        // 压敏修正 (摄动法)
        if (std::abs(gamaD) > 1e-9) {
            double arg = 1.0 - gamaD * pd_real;
            if (arg > 1e-12) pd_real = -1.0 / gamaD * std::log(arg);
        }
        outPD[k] = pd_real;
    };

    // 并行计算加速
    QtConcurrent::blockingMap(indexes, calculateSinglePoint);
}

// Laplace 空间解
double ModelSolver01_06::flaplace_composite(double z, const QMap<QString, double>& p) {
    // 读取参数
    double M12 = p.value("M12", 1.0);
    double L = p.value("L", 1000.0);
    double Lf = p.value("Lf", 100.0);
    double rm = p.value("rm", 500.0);
    double re = p.value("re", 20000.0);
    double LfD = (L > 1e-9) ? Lf / L : 0.1;
    double rmD = (L > 1e-9) ? rm / L : 0.5;
    double reD = (L > 1e-9) ? re / L : 20.0;
    double eta12 = p.value("eta12", 1.0);

    int n_fracs = (int)p.value("nf", 1);
    int n_seg = (int)p.value("n_seg", 10);
    double spacingD = (n_fracs > 1) ? 0.9 / (double)(n_fracs - 1) : 0.0;

    double fs1 = 1.0, fs2 = 1.0;

    int id = (int)m_type + 1; // 1-36

    // --- 介质函数定义 ---
    // Group 1 (1-12) & Group 2 (13-24): 内区为夹层型 (双重孔隙)
    if (id <= 24) {
        double omga1 = p.value("omega1", 0.4);
        double remda1 = p.value("lambda1", 1e-3);
        double one_minus = 1.0 - omga1;
        fs1 = (omga1 * one_minus * z + remda1) / (one_minus * z + remda1);
    } else {
        // Group 3 (25-36): 内区为均质
        fs1 = 1.0;
    }

    // Group 1 (1-12): 外区为夹层型 (双重孔隙)
    if (id <= 12) {
        double omga2 = p.value("omega2", 0.08);
        double remda2 = p.value("lambda2", 1e-4);
        double one_minus = 1.0 - omga2;
        fs2 = eta12 * (omga2 * one_minus * eta12 * z + remda2) / (one_minus * eta12 * z + remda2);
    } else {
        // Group 2 (13-24) & Group 3 (25-36): 外区为均质
        fs2 = eta12;
    }

    // 调用边界元计算
    double pf = PWD_composite(z, fs1, fs2, M12, LfD, rmD, reD, n_seg, n_fracs, spacingD, m_type);

    // --- 井储与表皮效应 ---
    // 0: Const, 1: Line, 2: Fair, 3: Hegeman
    int storageType = (int)m_type % 4;

    // 如果不是线源解 (即 Const, Fair, Hegeman)，在Laplace空间叠加基础的定井储和表皮
    // Fair/Hegeman 是在定井储基础上叠加的，所以这里先算好定井储
    if (storageType != 1) {
        double CD = p.value("cD", 0.0);
        double S = p.value("S", 0.0);
        if (CD > 1e-12 || std::abs(S) > 1e-12) {
            double num = z * pf + S;
            double den = z + CD * z * z * num;
            if (std::abs(den) > 1e-100) pf = num / den;
        }
    }
    // 线源解 (LineSource) 直接返回 pf，不叠加井储表皮

    return pf;
}

// 边界元核心求解
double ModelSolver01_06::PWD_composite(double z, double fs1, double fs2, double M12, double LfD, double rmD, double reD,
                                       int n_seg, int n_fracs, double spacingD, ModelType type) {
    // 确定边界类型
    int id = (int)type + 1;
    int groupIdx = (id - 1) % 12; // 0-11

    // 0-3: Infinite, 4-7: Closed, 8-11: ConstP
    bool isInfinite = (groupIdx < 4);
    bool isClosed = (groupIdx >= 4 && groupIdx < 8);
    bool isConstP = (groupIdx >= 8);

    // --- 构造 BEM 矩阵 ---
    int total_segments = n_fracs * n_seg;
    double segLen = 2.0 * LfD / n_seg;
    QVector<Point2D> segmentCenters;
    segmentCenters.reserve(total_segments);
    double startX = -(n_fracs - 1) * spacingD / 2.0;

    // 生成裂缝单元中心点
    for (int k = 0; k < n_fracs; ++k) {
        double currentX = startX + k * spacingD;
        for (int i = 0; i < n_seg; ++i) {
            double currentY = -LfD + (i + 0.5) * segLen;
            segmentCenters.append({currentX, currentY});
        }
    }

    double gama1 = sqrt(z * fs1);
    double gama2 = sqrt(z * fs2);
    double arg_g1_rm = gama1 * rmD;
    double arg_g2_rm = gama2 * rmD;

    // 计算 Bessel 函数值
    double k0_g2_rm = safe_bessel_k(0, arg_g2_rm);
    double k1_g2_rm = safe_bessel_k(1, arg_g2_rm);
    double k0_g1_rm = safe_bessel_k(0, arg_g1_rm);
    double k1_g1_rm = safe_bessel_k(1, arg_g1_rm);

    double term_mAB_i0 = 0.0;
    double term_mAB_i1 = 0.0;

    // 处理外边界条件
    if (!isInfinite && reD > 1e-5) {
        double arg_re = gama2 * reD;
        double i0_re_s = safe_bessel_i_scaled(0, arg_re);
        double i1_re_s = safe_bessel_i_scaled(1, arg_re);
        double k1_re = safe_bessel_k(1, arg_re);
        double k0_re = safe_bessel_k(0, arg_re);

        double i0_g2_rm_s = safe_bessel_i_scaled(0, arg_g2_rm);
        double i1_g2_rm_s = safe_bessel_i_scaled(1, arg_g2_rm);

        // 指数修正因子
        double exp_factor = 0.0;
        if ((arg_g2_rm - arg_re) > -700.0) exp_factor = std::exp(arg_g2_rm - arg_re);

        if (isClosed && i1_re_s > 1e-100) {
            term_mAB_i0 = (k1_re / i1_re_s) * i0_g2_rm_s * exp_factor;
            term_mAB_i1 = (k1_re / i1_re_s) * i1_g2_rm_s * exp_factor;
        } else if (isConstP && i0_re_s > 1e-100) {
            term_mAB_i0 = -(k0_re / i0_re_s) * i0_g2_rm_s * exp_factor;
            term_mAB_i1 = -(k0_re / i0_re_s) * i1_g2_rm_s * exp_factor;
        }
    }

    // 复合模型系数计算
    double term1 = term_mAB_i0 + k0_g2_rm;
    double term2 = term_mAB_i1 - k1_g2_rm;

    double Acup = M12 * gama1 * k1_g1_rm * term1 + gama2 * k0_g1_rm * term2;
    double i1_g1_rm_s = safe_bessel_i_scaled(1, arg_g1_rm);
    double i0_g1_rm_s = safe_bessel_i_scaled(0, arg_g1_rm);

    double Acdown_scaled = M12 * gama1 * i1_g1_rm_s * term1 - gama2 * i0_g1_rm_s * term2;
    if (std::abs(Acdown_scaled) < 1e-100) Acdown_scaled = (Acdown_scaled >= 0 ? 1e-100 : -1e-100);

    double Ac_prefactor = Acup / Acdown_scaled;

    // 填充系数矩阵
    int size = total_segments + 1;
    Eigen::MatrixXd A_mat(size, size);
    Eigen::VectorXd b_vec(size);
    b_vec.setZero();
    b_vec(total_segments) = 1.0;

    double halfLen = segLen / 2.0;

    for (int i = 0; i < total_segments; ++i) {
        for (int j = i; j < total_segments; ++j) {
            Point2D pi = segmentCenters[i];
            Point2D pj = segmentCenters[j];

            double dx_sq = (pi.x - pj.x) * (pi.x - pj.x);

            // 积分核函数
            auto integrand = [&](double a) -> double {
                double dy = pi.y - (pj.y + a);
                double dist_val = std::sqrt(dx_sq + dy * dy);
                double arg_dist = gama1 * dist_val;
                double term2_val = 0.0;
                double exponent = arg_dist - arg_g1_rm;
                if (exponent > -700.0) term2_val = Ac_prefactor * safe_bessel_i_scaled(0, arg_dist) * std::exp(exponent);
                return safe_bessel_k(0, arg_dist) + term2_val;
            };

            double val = 0.0;
            // 奇异积分处理
            if (i == j) val = 2.0 * adaptiveGauss(integrand, 0.0, halfLen, 1e-6, 0, 8);
            else if (std::abs(pi.x - pj.x) < 1e-9) val = adaptiveGauss(integrand, -halfLen, halfLen, 1e-6, 0, 5);
            else val = adaptiveGauss(integrand, -halfLen, halfLen, 1e-5, 0, 3);

            double element = val / (M12 * 2.0 * LfD);
            A_mat(i, j) = element;
            if (i != j) A_mat(j, i) = element;
        }
    }

    // 边界条件: 定压
    for (int i = 0; i < total_segments; ++i) {
        A_mat(i, total_segments) = -1.0;
        A_mat(total_segments, i) = z;
    }
    A_mat(total_segments, total_segments) = 0.0;

    // 求解线性方程组
    Eigen::VectorXd x_sol = A_mat.partialPivLu().solve(b_vec);
    return x_sol(total_segments);
}

// --- 辅助函数实现 ---
double ModelSolver01_06::scaled_besseli(int v, double x) { return safe_bessel_i_scaled(v, x); }

double ModelSolver01_06::gauss15(std::function<double(double)> f, double a, double b) {
    static const double X[] = { 0.0, 0.20119409, 0.39415135, 0.57097217, 0.72441773, 0.84820658, 0.93729853, 0.98799252 };
    static const double W[] = { 0.20257824, 0.19843149, 0.18616100, 0.16626921, 0.13957068, 0.10715922, 0.07036605, 0.03075324 };
    double h = 0.5 * (b - a);
    double c = 0.5 * (a + b);
    double s = W[0] * f(c);
    for (int i = 1; i < 8; ++i) {
        double dx = h * X[i];
        s += W[i] * (f(c - dx) + f(c + dx));
    }
    return s * h;
}

double ModelSolver01_06::adaptiveGauss(std::function<double(double)> f, double a, double b, double eps, int depth, int maxDepth) {
    double c = (a + b) / 2.0;
    double v1 = gauss15(f, a, b);
    double v2 = gauss15(f, a, c) + gauss15(f, c, b);
    if (depth >= maxDepth || std::abs(v1 - v2) < eps * (std::abs(v2) + 1.0)) return v2;
    return adaptiveGauss(f, a, c, eps/2, depth+1, maxDepth) + adaptiveGauss(f, c, b, eps/2, depth+1, maxDepth);
}

void ModelSolver01_06::precomputeStehfestCoeffs(int N) {
    if (m_currentN == N && !m_stehfestCoeffs.isEmpty()) return;
    m_currentN = N; m_stehfestCoeffs.resize(N + 1);
    for (int i = 1; i <= N; ++i) {
        double s = 0.0;
        int k1 = (i + 1) / 2;
        int k2 = std::min(i, N / 2);
        for (int k = k1; k <= k2; ++k) {
            double num = std::pow((double)k, N / 2.0) * factorial(2 * k);
            double den = factorial(N / 2 - k) * factorial(k) * factorial(k - 1) * factorial(i - k) * factorial(2 * k - i);
            if (den != 0) s += num / den;
        }
        double sign = ((i + N / 2) % 2 == 0) ? 1.0 : -1.0;
        m_stehfestCoeffs[i] = sign * s;
    }
}

double ModelSolver01_06::getStehfestCoeff(int i, int N) {
    if (m_currentN != N) return 0.0;
    if (i < 1 || i > N) return 0.0;
    return m_stehfestCoeffs[i];
}

double ModelSolver01_06::factorial(int n) {
    if(n <= 1) return 1.0;
    double r = 1.0;
    for(int i = 2; i <= n; ++i) r *= i;
    return r;
}
