/*
 * 文件名: modelsolver2.h
 * 文件作用: 求解器2 - 负责页岩型复合模型 (ID 1-36，全局ID对应37-72)
 * 功能描述:
 * 1. 涵盖模型范围：
 * - Model 1-12: 页岩型+页岩型
 * - Model 13-24: 页岩型+均质
 * - Model 25-36: 页岩型+双重孔隙
 * 2. 井储模型支持：定井储、线源解、Fair、Hegeman。
 * 3. 算法核心：瞬态平板模型 (Transient Slab) + 边界元。
 */

#ifndef MODELSOLVER2_H
#define MODELSOLVER2_H

#include <QMap>
#include <QVector>
#include <QString>
#include <tuple>
#include <functional>
#include <QtConcurrent>

using ModelCurveData = std::tuple<QVector<double>, QVector<double>, QVector<double>>;

class ModelSolver2
{
public:
    enum ModelType {
        // === Group 1: 页岩型 + 页岩型 (1-12) ===
        // 无限大
        Model_1 = 0, Model_2, Model_3, Model_4,
        // 封闭
        Model_5, Model_6, Model_7, Model_8,
        // 定压
        Model_9, Model_10, Model_11, Model_12,

        // === Group 2: 页岩型 + 均质 (13-24) ===
        Model_13, Model_14, Model_15, Model_16,
        Model_17, Model_18, Model_19, Model_20,
        Model_21, Model_22, Model_23, Model_24,

        // === Group 3: 页岩型 + 双重孔隙 (25-36) ===
        Model_25, Model_26, Model_27, Model_28,
        Model_29, Model_30, Model_31, Model_32,
        Model_33, Model_34, Model_35, Model_36
    };

    explicit ModelSolver2(ModelType type);
    virtual ~ModelSolver2();

    void setHighPrecision(bool high);
    ModelCurveData calculateTheoreticalCurve(const QMap<QString, double>& params, const QVector<double>& providedTime = QVector<double>());
    static QString getModelName(ModelType type, bool verbose = true);
    static QVector<double> generateLogTimeSteps(int count, double startExp, double endExp);

private:
    void calculatePDandDeriv(const QVector<double>& tD, const QMap<QString, double>& params,
                             std::function<double(double, const QMap<QString, double>&)> laplaceFunc,
                             QVector<double>& outPD, QVector<double>& outDeriv);

    double flaplace_composite(double z, const QMap<QString, double>& p);
    double PWD_composite(double z, double fs1, double fs2, double M12, double LfD, double rmD, double reD,
                         int n_seg, int n_fracs, double spacingD, ModelType type);

    double calc_fs_dual(double u, double omega, double lambda);
    double calc_fs_shale(double u, double omega, double lambda); // 页岩平板模型

    double scaled_besseli(int v, double x);
    double gauss15(std::function<double(double)> f, double a, double b);
    double adaptiveGauss(std::function<double(double)> f, double a, double b, double eps, int depth, int maxDepth);
    double getStehfestCoeff(int i, int N);
    void precomputeStehfestCoeffs(int N);
    double factorial(int n);

private:
    ModelType m_type;
    bool m_highPrecision;
    QVector<double> m_stehfestCoeffs;
    int m_currentN;
};

#endif // MODELSOLVER2_H
