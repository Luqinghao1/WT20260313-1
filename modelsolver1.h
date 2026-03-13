/*
 * 文件名: modelsolver1.h
 * 文件作用: 求解器1 - 负责夹层型及径向复合模型 (ID 1-36)
 * 功能描述:
 * 1. 涵盖模型范围：
 * - Model 1-12: 夹层型+夹层型 (内区夹层，外区夹层)
 * - Model 13-24: 夹层型+均质 (内区夹层，外区均质)
 * - Model 25-36: 径向复合 (内区均质，外区均质)
 * 2. 井储模型支持：
 * - 定井储 (Constant): 基础模型，包含 Cd 和 S。
 * - 线源解 (LineSource): 不含 Cd 和 S，纯地层响应。
 * - Fair模型 (Fair): 基于定井储，叠加变井储时间因子 (1-exp)。
 * - Hegeman模型 (Hegeman): 基于定井储，叠加变井储时间因子 (erf)。
 * 3. 算法核心：
 * - Stehfest 数值反演算法。
 * - 复合油藏边界元解法 (PWD_composite)。
 * - 双重孔隙/夹层介质传输函数。
 */

#ifndef MODELSOLVER1_H
#define MODELSOLVER1_H

#include <QMap>
#include <QVector>
#include <QString>
#include <tuple>
#include <functional>
#include <QtConcurrent>

// 返回类型: <时间(t), 压力(Dp), 导数(Dp')>
using ModelCurveData = std::tuple<QVector<double>, QVector<double>, QVector<double>>;

class ModelSolver1
{
public:
    // 模型ID枚举 (1-36)
    // 分组逻辑: 3种介质组合 * 3种边界 * 4种井储
    enum ModelType {
        // === 第一组: 夹层型 + 夹层型 (1-12) ===
        // 无限大外边界
        Model_1 = 0, // 定井储
        Model_2,     // 线源解
        Model_3,     // Fair模型
        Model_4,     // Hegeman模型
        // 封闭边界
        Model_5, Model_6, Model_7, Model_8,
        // 定压边界
        Model_9, Model_10, Model_11, Model_12,

        // === 第二组: 夹层型 + 均质 (13-24) ===
        // 无限大
        Model_13, Model_14, Model_15, Model_16,
        // 封闭
        Model_17, Model_18, Model_19, Model_20,
        // 定压
        Model_21, Model_22, Model_23, Model_24,

        // === 第三组: 径向复合(均质+均质) (25-36) ===
        // 无限大
        Model_25, Model_26, Model_27, Model_28,
        // 封闭
        Model_29, Model_30, Model_31, Model_32,
        // 定压
        Model_33, Model_34, Model_35, Model_36
    };

    explicit ModelSolver1(ModelType type);
    virtual ~ModelSolver1();

    // 设置是否启用高精度Stehfest(N=10 vs N=6)
    void setHighPrecision(bool high);

    // 计算理论曲线主入口
    ModelCurveData calculateTheoreticalCurve(const QMap<QString, double>& params, const QVector<double>& providedTime = QVector<double>());

    // 获取模型显示名称
    static QString getModelName(ModelType type, bool verbose = true);

    // 辅助: 生成对数时间步
    static QVector<double> generateLogTimeSteps(int count, double startExp, double endExp);

private:
    // 计算无因次压力及导数 (Stehfest反演循环)
    void calculatePDandDeriv(const QVector<double>& tD, const QMap<QString, double>& params,
                             std::function<double(double, const QMap<QString, double>&)> laplaceFunc,
                             QVector<double>& outPD, QVector<double>& outDeriv);

    // Laplace空间主函数：处理介质、内外区耦合及井储表皮
    double flaplace_composite(double z, const QMap<QString, double>& p);

    // 边界元核心算法：求解复合油藏压力响应
    double PWD_composite(double z, double fs1, double fs2, double M12, double LfD, double rmD, double reD,
                         int n_seg, int n_fracs, double spacingD, ModelType type);

    // 介质函数 f(s) 计算
    double calc_fs_dual(double u, double omega, double lambda);

    // 数学工具函数
    double scaled_besseli(int v, double x);
    double gauss15(std::function<double(double)> f, double a, double b);
    double adaptiveGauss(std::function<double(double)> f, double a, double b, double eps, int depth, int maxDepth);

    // Stehfest 系数计算
    double getStehfestCoeff(int i, int N);
    void precomputeStehfestCoeffs(int N);
    double factorial(int n);

private:
    ModelType m_type;
    bool m_highPrecision;
    QVector<double> m_stehfestCoeffs;
    int m_currentN;
};

#endif // MODELSOLVER1_H
