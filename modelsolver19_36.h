/*
 * 文件名: modelsolver19_36.h
 * 文件作用: 压裂水平井页岩型复合模型 Group 2 (Model 37-72) 核心计算类头文件
 * 功能描述:
 * 1. 负责计算 modelsolver2.csv 中定义的36个页岩型试井模型。
 * 2. 涵盖三种储层组合：
 * - 页岩型 + 页岩型   (Model 2-1 到 2-12)
 * - 页岩型 + 均质     (Model 2-13 到 2-24)
 * - 页岩型 + 双重孔隙 (Model 2-25 到 2-36)
 * 3. 实现了瞬态平板模型(Transient Slab Matrix)用于描述页岩型介质。
 * 4. 支持 Fair 和 Hegeman 模型的变井储效应。
 */

#ifndef MODELSOLVER19_36_H
#define MODELSOLVER19_36_H

#include <QMap>
#include <QVector>
#include <QString>
#include <tuple>
#include <functional>
#include <QtConcurrent>

using ModelCurveData = std::tuple<QVector<double>, QVector<double>, QVector<double>>;

class ModelSolver19_36
{
public:
    // 模型类型枚举 (对应 modelsolver2.csv 中的 model2-1 到 model2-36)
    // 内部索引 0-35 (对应全局模型 ID 37-72)
    enum ModelType {
        // --- 第一组: 页岩型 + 页岩型 ---
        // 无限大 (Const, Line, Fair, Hegeman)
        Model_1 = 0, Model_2, Model_3, Model_4,
        // 封闭
        Model_5, Model_6, Model_7, Model_8,
        // 定压
        Model_9, Model_10, Model_11, Model_12,

        // --- 第二组: 页岩型 + 均质 ---
        Model_13, Model_14, Model_15, Model_16, // 无限大
        Model_17, Model_18, Model_19, Model_20, // 封闭
        Model_21, Model_22, Model_23, Model_24, // 定压

        // --- 第三组: 页岩型 + 双重孔隙 ---
        Model_25, Model_26, Model_27, Model_28, // 无限大
        Model_29, Model_30, Model_31, Model_32, // 封闭
        Model_33, Model_34, Model_35, Model_36  // 定压
    };

    explicit ModelSolver19_36(ModelType type);
    virtual ~ModelSolver19_36();

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

    // 介质函数
    double calc_fs_dual(double u, double omega, double lambda); // 双孔
    double calc_fs_shale(double u, double omega, double lambda); // 页岩 (Transient Slab)

    // 数学辅助
    double scaled_besseli(int v, double x);
    double gauss15(std::function<double(double)> f, double a, double b);
    double adaptiveGauss(std::function<double(double)> f, double a, double b, double eps, int depth, int maxDepth);

    // Stehfest
    double getStehfestCoeff(int i, int N);
    void precomputeStehfestCoeffs(int N);
    double factorial(int n);

private:
    ModelType m_type;
    bool m_highPrecision;
    QVector<double> m_stehfestCoeffs;
    int m_currentN;
};

#endif // MODELSOLVER19_36_H
