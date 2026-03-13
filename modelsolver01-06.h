/*
 * 文件名: modelsolver01-06.h
 * 文件作用: 压裂水平井复合模型 Group 1 (Model 1-36) 核心计算类头文件
 * 功能描述:
 * 1. 负责计算 modelsolver1.csv 中定义的36个试井模型。
 * 2. 涵盖三种储层组合：
 * - 夹层型 + 夹层型 (Model 1-12)
 * - 夹层型 + 均质   (Model 13-24)
 * - 径向复合 (均质+均质) (Model 25-36)
 * 3. 每个储层组合下包含3种边界(无限大、封闭、定压)和4种井储类型(定井储、线源解、Fair、Hegeman)。
 * 4. 提供了基于 Stehfest 数值反演和边界元方法(BEM)的核心求解算法。
 */

#ifndef MODELSOLVER01_06_H
#define MODELSOLVER01_06_H

#include <QMap>
#include <QVector>
#include <QString>
#include <tuple>
#include <functional>
#include <QtConcurrent>

// 类型定义: <时间序列(t), 压力序列(Dp), 导数序列(Dp')>
using ModelCurveData = std::tuple<QVector<double>, QVector<double>, QVector<double>>;

class ModelSolver01_06
{
public:
    // 模型类型枚举 (对应 modelsolver1.csv 中的36个模型)
    // 命名规则: Model_序号 (1-36)
    enum ModelType {
        // --- 第一组: 夹层型 + 夹层型 (1-12) ---
        // 无限大边界 (4种井储)
        Model_1 = 0, Model_2, Model_3, Model_4,
        // 封闭边界 (4种井储)
        Model_5, Model_6, Model_7, Model_8,
        // 定压边界 (4种井储)
        Model_9, Model_10, Model_11, Model_12,

        // --- 第二组: 夹层型 + 均质 (13-24) ---
        // 无限大边界
        Model_13, Model_14, Model_15, Model_16,
        // 封闭边界
        Model_17, Model_18, Model_19, Model_20,
        // 定压边界
        Model_21, Model_22, Model_23, Model_24,

        // --- 第三组: 径向复合(均质+均质) (25-36) ---
        // 无限大边界
        Model_25, Model_26, Model_27, Model_28,
        // 封闭边界
        Model_29, Model_30, Model_31, Model_32,
        // 定压边界
        Model_33, Model_34, Model_35, Model_36
    };

    /**
     * @brief 构造函数
     * @param type 模型类型 (Model_1 到 Model_36)
     */
    explicit ModelSolver01_06(ModelType type);
    virtual ~ModelSolver01_06();

    /**
     * @brief 设置高精度计算模式
     * @param high true为高精度(Stehfest N=10/12)，false为低精度(N=6/8)
     */
    void setHighPrecision(bool high);

    /**
     * @brief 计算理论曲线的核心接口
     * @param params 模型参数集合 (包含 k, S, C, alpha, C_phi 等)
     * @param providedTime 指定的时间序列 (若为空则自动生成)
     * @return 包含时间、压力、导数的元组
     */
    ModelCurveData calculateTheoreticalCurve(const QMap<QString, double>& params, const QVector<double>& providedTime = QVector<double>());

    /**
     * @brief 获取模型的中文名称和描述
     * @param type 模型类型
     * @param verbose 是否包含详细描述(井储、边界、介质)
     */
    static QString getModelName(ModelType type, bool verbose = true);

    /**
     * @brief 生成对数分布的时间步
     */
    static QVector<double> generateLogTimeSteps(int count, double startExp, double endExp);

private:
    // 计算无因次压力(PD)和导数
    void calculatePDandDeriv(const QVector<double>& tD, const QMap<QString, double>& params,
                             std::function<double(double, const QMap<QString, double>&)> laplaceFunc,
                             QVector<double>& outPD, QVector<double>& outDeriv);

    // Laplace空间下的复合模型解函数
    double flaplace_composite(double z, const QMap<QString, double>& p);

    // 边界元方法(BEM)求解核心函数
    // 求解线性方程组 Ax=b 得到压力解
    double PWD_composite(double z, double fs1, double fs2, double M12, double LfD, double rmD, double reD,
                         int n_seg, int n_fracs, double spacingD, ModelType type);

    // --- 数学辅助函数 ---
    double scaled_besseli(int v, double x); // 标度修正的Bessel I函数
    double gauss15(std::function<double(double)> f, double a, double b); // Gauss-Kronrod 积分
    double adaptiveGauss(std::function<double(double)> f, double a, double b, double eps, int depth, int maxDepth); // 自适应积分

    // --- Stehfest 反演算法辅助 ---
    double getStehfestCoeff(int i, int N);
    void precomputeStehfestCoeffs(int N);
    double factorial(int n);

private:
    ModelType m_type;
    bool m_highPrecision;
    QVector<double> m_stehfestCoeffs;
    int m_currentN;
};

#endif // MODELSOLVER01_06_H
