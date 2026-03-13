/*
 * 文件名: modelsolver1.cpp
 * 文件作用: 求解器1实现文件
 * 功能描述:
 * 1. 实现了 Model 1-36 的完整计算逻辑。
 * 2. 修正了 Fair/Hegeman 模型的计算逻辑：
 * - Laplace 空间：包含常数井储(Cd)和表皮(S)。
 * - 时间域：通过修改 p_coeff 叠加变井储效应 (alpha, C_phi)。
 * 3. 修正了线源解逻辑：Laplace 空间不包含 Cd 和 S。
 */

#include "modelsolver1.h"
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

// --- 静态辅助函数 ---

// 安全的 Bessel K 函数 (防止下溢)
static double safe_bessel_k(int v, double x) {
    if (x < 1e-15) x = 1e-15;
    try { return boost::math::cyl_bessel_k(v, x); } catch (...) { return 0.0; }
}

// 缩放的 Bessel I 函数 (防止上溢, 返回 I(x)*exp(-x))
static double safe_bessel_i_scaled(int v, double x) {
    if (x < 0) x = -x;
    if (x > 600.0) return 1.0 / std::sqrt(2.0 * M_PI * x);
    try { return boost::math::cyl_bessel_i(v, x) * std::exp(-x); } catch (...) { return 0.0; }
}

struct Point2D { double x; double y; };

// --- 构造与析构 ---

ModelSolver1::ModelSolver1(ModelType type)
    : m_type(type), m_highPrecision(true), m_currentN(0) {
    precomputeStehfestCoeffs(10); // 默认N=10
}

ModelSolver1::~ModelSolver1() {}

void ModelSolver1::setHighPrecision(bool high) { m_highPrecision = high; }

// --- 模型名称生成 ---

QString ModelSolver1::getModelName(ModelType type, bool verbose) {
    int id = (int)type + 1;
    QString base;

    // 确定基础模型名称
    if (id <= 12) base = QString("夹层型储层试井解释%1").arg(id);
    else if (id <= 24) base = QString("夹层型储层试井解释%1").arg(id);
    else base = QString("径向复合储层试井解释%1").arg(id);

    if (!verbose) return base;

    // 确定井储类型 (0:Constant, 1:Line, 2:Fair, 3:Hegeman)
    int storeRem = (id - 1) % 4;
    QString strStore;
    if (storeRem == 0) strStore = "定井储";
    else if (storeRem == 1) strStore = "线源解";
    else if (storeRem == 2) strStore = "Fair模型";
    else strStore = "Hegeman模型";

    return QString("%1 (%2)").arg(base).arg(strStore);
}

// --- 时间步生成 ---

QVector<double> ModelSolver1::generateLogTimeSteps(int count, double startExp, double endExp) {
    QVector<double> t;
    if (count <= 0) return t;
    t.reserve(count);
    for (int i = 0; i < count; ++i) {
        t.append(pow(10.0, startExp + (endExp - startExp) * i / (count - 1)));
    }
    return t;
}

// --- 核心计算流程 ---

ModelCurveData ModelSolver1::calculateTheoreticalCurve(const QMap<QString, double>& params, const QVector<double>& providedTime) {
    QVector<double> tPoints = providedTime;
    if (tPoints.isEmpty()) tPoints = generateLogTimeSteps(100, -3.0, 3.0);

    // 1. 获取物理参数
    double phi = params.value("phi", 0.05);
    double mu = params.value("mu", 0.5);
    double B = params.value("B", 1.05);
    double Ct = params.value("Ct", 5e-4);
    double q = params.value("q", 5.0);
    double h = params.value("h", 20.0);
    double kf = params.value("kf", 1e-3);
    double L = params.value("L", 1000.0);
    if (L < 1e-9) L = 1000.0;

    // Fair/Hegeman 模型特有参数
    double alpha = params.value("alpha", 1e-1);
    double C_phi = params.value("C_phi", 1e-4);

    // 2. 计算无因次时间系数
    double td_coeff = 14.4 * kf / (phi * mu * Ct * pow(L, 2));
    QVector<double> tD_vec;
    tD_vec.reserve(tPoints.size());
    for(double t : tPoints) tD_vec.append(td_coeff * t);

    // 3. 准备 Stehfest 反演参数
    QMap<QString, double> calcParams = params;
    int N = (int)calcParams.value("N", 10);
    if (N < 4 || N > 18 || N % 2 != 0) N = 10;
    calcParams["N"] = N;
    precomputeStehfestCoeffs(N);

    if (!calcParams.contains("n_seg")) calcParams["n_seg"] = 5;

    // 4. 执行 Laplace 反演计算 P_D (包含定井储效应)
    QVector<double> PD_vec, Deriv_vec;
    auto func = std::bind(&ModelSolver1::flaplace_composite, this, std::placeholders::_1, std::placeholders::_2);
    calculatePDandDeriv(tD_vec, calcParams, func, PD_vec, Deriv_vec);

    // 5. 计算基础压力系数 (单位转换)
    double p_coeff_base = 1.842e-3 * q * mu * B / (kf * h);

    QVector<double> finalP(tPoints.size()), finalDP(tPoints.size());

    // 6. 应用变井储效应 (Fair/Hegeman)
    // storeRem: 0=Constant, 1=Line, 2=Fair, 3=Hegeman
    int storeRem = ((int)m_type) % 4;

    for(int i=0; i<tPoints.size(); ++i) {
        double t = tPoints[i];
        double current_p_coeff = p_coeff_base; // 默认系数

        // Fair模型修正: 增加 C_phi * (1 - exp(-t/alpha))
        if (storeRem == 2) {
            current_p_coeff += C_phi * (1.0 - std::exp(-t / alpha));
        }
        // Hegeman模型修正: 增加 C_phi * erf(t/alpha)
        else if (storeRem == 3) {
            current_p_coeff += C_phi * std::erf(t / alpha);
        }
        // 定井储(0) 和 线源解(1) 不修改 p_coeff

        finalP[i] = current_p_coeff * PD_vec[i];
        finalDP[i] = current_p_coeff * Deriv_vec[i];
    }

    return std::make_tuple(tPoints, finalP, finalDP);
}

// --- Stehfest 数值反演 ---

void ModelSolver1::calculatePDandDeriv(const QVector<double>& tD, const QMap<QString, double>& params,
                                       std::function<double(double, const QMap<QString, double>&)> laplaceFunc,
                                       QVector<double>& outPD, QVector<double>& outDeriv) {
    int numPoints = tD.size();
    outPD.resize(numPoints);
    outDeriv.resize(numPoints);
    int N = (int)params.value("N", 10);
    double ln2 = 0.6931471805599453;

    QVector<int> indexes(numPoints);
    std::iota(indexes.begin(), indexes.end(), 0);

    auto calculateSinglePoint = [&](int k) {
        double t = tD[k];
        if (t <= 1e-10) { outPD[k] = 0.0; return; }
        double pd_val = 0.0;
        for (int m = 1; m <= N; ++m) {
            double z = m * ln2 / t;
            double pf = laplaceFunc(z, params);
            pd_val += getStehfestCoeff(m, N) * pf;
        }
        outPD[k] = pd_val * ln2 / t;
    };

    // 并行计算加速
    QtConcurrent::blockingMap(indexes, calculateSinglePoint);

    // 计算导数
    if (numPoints > 2) {
        outDeriv = PressureDerivativeCalculator::calculateBourdetDerivative(tD, outPD, 0.1);
    } else {
        outDeriv.fill(0.0);
    }
}

// --- Laplace 空间解 ---

double ModelSolver1::flaplace_composite(double z, const QMap<QString, double>& p) {
    // 1. 获取并归一化几何参数
    double M12 = p.value("M12", 1.0);
    double eta12 = p.value("eta", 0.2); if(p.contains("eta12")) eta12 = p.value("eta12");

    double L = p.value("L", 1000.0);
    double Lf = p.value("Lf", 100.0);
    double rm = p.value("rm", 500.0);
    double re = p.value("re", 20000.0);

    double LfD = (L > 1e-9) ? Lf / L : 0.1;
    double rmD = (L > 1e-9) ? rm / L : 0.5;
    double reD = (L > 1e-9) ? re / L : 20.0;

    int n_fracs = (int)p.value("nf", 1);
    int n_seg = (int)p.value("n_seg", 5);
    double spacingD = (n_fracs > 1) ? 0.9 / (double)(n_fracs - 1) : 0.0;

    // 2. 计算介质函数 f(s)
    double fs1 = 1.0;
    double fs2 = 1.0;
    int id = (int)m_type + 1; // 1-36

    // Model 1-12: 夹层(内) + 夹层(外)
    if (id <= 12) {
        double w1 = p.value("omega1", 0.4); double l1 = p.value("lambda1", 1e-3);
        double w2 = p.value("omega2", 0.08); double l2 = p.value("lambda2", 1e-4);
        fs1 = calc_fs_dual(z, w1, l1);
        fs2 = eta12 * calc_fs_dual(eta12 * z, w2, l2);
    }
    // Model 13-24: 夹层(内) + 均质(外)
    else if (id <= 24) {
        double w1 = p.value("omega1", 0.4); double l1 = p.value("lambda1", 1e-3);
        fs1 = calc_fs_dual(z, w1, l1);
        fs2 = eta12;
    }
    // Model 25-36: 均质(内) + 均质(外)
    else {
        fs1 = 1.0;
        fs2 = eta12;
    }

    // 3. 计算地层压力响应 P_D (边界元解)
    double pf = PWD_composite(z, fs1, fs2, M12, LfD, rmD, reD, n_seg, n_fracs, spacingD, m_type);

    // 4. 应用井储和表皮
    // 逻辑修正:
    // 0 (Constant): 应用 Cd, S
    // 1 (Line): 不应用 (纯地层)
    // 2 (Fair): 应用 Cd, S (在定井储基础上叠加变井储)
    // 3 (Hegeman): 应用 Cd, S (在定井储基础上叠加变井储)

    int storeRem = ((int)m_type) % 4;

    if (storeRem != 1) { // 只要不是线源解，都应用井储表皮
        double CD = p.value("cD", 0.0);
        double S = p.value("S", 0.0);
        if (CD > 1e-12 || std::abs(S) > 1e-12) {
            double num = z * pf + S;
            double den = z + CD * z * z * num;
            if (std::abs(den) > 1e-100) pf = num / den;
        }
    }

    return pf;
}

// 边界元求解器 (保持原逻辑，适配边界索引)
double ModelSolver1::PWD_composite(double z, double fs1, double fs2, double M12, double LfD, double rmD, double reD,
                                   int n_seg, int n_fracs, double spacingD, ModelType type) {
    // 边界类型判断 (每组12个: 0-3无限, 4-7封闭, 8-11定压)
    int groupIdx = ((int)type) % 12;
    bool isInf = (groupIdx < 4);
    bool isClosed = (groupIdx >= 4 && groupIdx < 8);
    bool isConstP = (groupIdx >= 8);

    // --- 边界元矩阵构建与求解 (同原代码) ---
    int total_segments = n_fracs * n_seg;
    double segLen = 2.0 * LfD / n_seg;
    QVector<Point2D> segmentCenters;
    double startX = -(n_fracs - 1) * spacingD / 2.0;

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

    double k0_g2_rm = safe_bessel_k(0, arg_g2_rm);
    double k1_g2_rm = safe_bessel_k(1, arg_g2_rm);
    double k0_g1_rm = safe_bessel_k(0, arg_g1_rm);
    double k1_g1_rm = safe_bessel_k(1, arg_g1_rm);

    double term_mAB_i0 = 0.0;
    double term_mAB_i1 = 0.0;

    if (!isInf && reD > 1e-5) {
        double arg_re = gama2 * reD;
        double i0_re_s = safe_bessel_i_scaled(0, arg_re);
        double i1_re_s = safe_bessel_i_scaled(1, arg_re);
        double k1_re = safe_bessel_k(1, arg_re);
        double k0_re = safe_bessel_k(0, arg_re);
        double i0_g2_rm_s = safe_bessel_i_scaled(0, arg_g2_rm);
        double i1_g2_rm_s = safe_bessel_i_scaled(1, arg_g2_rm);

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

    double term1 = term_mAB_i0 + k0_g2_rm;
    double term2 = term_mAB_i1 - k1_g2_rm;
    double Acup = M12 * gama1 * k1_g1_rm * term1 + gama2 * k0_g1_rm * term2;
    double i1_g1_rm_s = safe_bessel_i_scaled(1, arg_g1_rm);
    double i0_g1_rm_s = safe_bessel_i_scaled(0, arg_g1_rm);
    double Acdown_scaled = M12 * gama1 * i1_g1_rm_s * term1 - gama2 * i0_g1_rm_s * term2;
    if (std::abs(Acdown_scaled) < 1e-100) Acdown_scaled = (Acdown_scaled >= 0 ? 1e-100 : -1e-100);
    double Ac_prefactor = Acup / Acdown_scaled;

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
            if (i == j) val = 2.0 * adaptiveGauss(integrand, 0.0, halfLen, 1e-6, 0, 8);
            else if (std::abs(pi.x - pj.x) < 1e-9) val = adaptiveGauss(integrand, -halfLen, halfLen, 1e-6, 0, 5);
            else val = adaptiveGauss(integrand, -halfLen, halfLen, 1e-5, 0, 3);
            double element = val / (M12 * 2.0 * LfD);
            A_mat(i, j) = element;
            if (i != j) A_mat(j, i) = element;
        }
    }
    for (int i = 0; i < total_segments; ++i) {
        A_mat(i, total_segments) = -1.0;
        A_mat(total_segments, i) = z;
    }
    A_mat(total_segments, total_segments) = 0.0;
    Eigen::VectorXd x_sol = A_mat.partialPivLu().solve(b_vec);
    return x_sol(total_segments);
}

// 夹层型/双重孔隙介质函数
double ModelSolver1::calc_fs_dual(double u, double omega, double lambda) {
    double one_minus = 1.0 - omega;
    double den = one_minus * u + lambda;
    if (std::abs(den) < 1e-20) return 0.0;
    return (omega * one_minus * u + lambda) / den;
}

// 数学辅助函数
void ModelSolver1::precomputeStehfestCoeffs(int N) {
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
double ModelSolver1::getStehfestCoeff(int i, int N) { return (i < 1 || i > N) ? 0.0 : m_stehfestCoeffs[i]; }
double ModelSolver1::factorial(int n) { double r = 1.0; for(int i=2; i<=n; ++i) r*=i; return r; }
double ModelSolver1::scaled_besseli(int v, double x) { return safe_bessel_i_scaled(v, x); }
double ModelSolver1::gauss15(std::function<double(double)> f, double a, double b) {
    static const double X[] = { 0.0, 0.20119409, 0.39415135, 0.57097217, 0.72441773, 0.84820658, 0.93729853, 0.98799252 };
    static const double W[] = { 0.20257824, 0.19843149, 0.18616100, 0.16626921, 0.13957068, 0.10715922, 0.07036605, 0.03075324 };
    double h = 0.5 * (b - a); double c = 0.5 * (a + b); double s = W[0] * f(c);
    for (int i = 1; i < 8; ++i) { double dx = h * X[i]; s += W[i] * (f(c - dx) + f(c + dx)); }
    return s * h;
}
double ModelSolver1::adaptiveGauss(std::function<double(double)> f, double a, double b, double eps, int depth, int maxDepth) {
    double c = (a + b) / 2.0; double v1 = gauss15(f, a, b); double v2 = gauss15(f, a, c) + gauss15(f, c, b);
    if (depth >= maxDepth || std::abs(v1 - v2) < eps * (std::abs(v2) + 1.0)) return v2;
    return adaptiveGauss(f, a, c, eps/2, depth+1, maxDepth) + adaptiveGauss(f, c, b, eps/2, depth+1, maxDepth);
}
