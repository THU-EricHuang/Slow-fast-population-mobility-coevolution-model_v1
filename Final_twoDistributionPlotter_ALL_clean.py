import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import math


class LogLogPDFPlotter:
    def __init__(self, df, filename):
        self.df = df
        self.filename = filename

    def plot_log_log_pdf(
        self,
        path,
        param_name,
        column_name,
        plot_type="weight",
        plot_ss=True,
        ss_df=None,
        ss_column_name=None,
        use_bounds=False,
        min_exp=None,
        max_exp=None,
        city="GZ",
    ):
        plt.figure(figsize=(12, 7))

        # ---------- ① 自动确定对数区间 ----------
        if not use_bounds:
            min_val = self.df[column_name].min()
            max_val = self.df[column_name].max()

            if plot_ss and ss_df is not None and ss_column_name is not None:
                ss_min_val = ss_df[ss_column_name].min()
                ss_max_val = ss_df[ss_column_name].max()
                min_val = min(min_val, ss_min_val)
                max_val = max(max_val, ss_max_val)

            min_exp = math.floor(np.log10(min_val)) if min_val > 0 else 0
            max_exp = math.ceil(np.log10(max_val)) if max_val > 0 else 6

        if use_bounds:
            min_exp = 0 if min_exp is None else min_exp
            max_exp = 6 if max_exp is None else max_exp

        # ---------- ② 对数分箱 ----------
        bins = np.logspace(min_exp, max_exp, int((max_exp - min_exp) / 0.1) + 1)
        bin_widths = np.diff(bins)

        # ---------- ③ Model 数据 ----------
        mask_m = self.df[column_name].between(10 ** min_exp, 10 ** max_exp)
        counts_m, _ = np.histogram(self.df.loc[mask_m, column_name], bins=bins, density=True)
        # area_m = np.sum(counts_m * bin_widths)
        # print(f"[{self.filename}] model PDF area  = {area_m:.6f}")

        # ---------- ④ Empirical 数据 ----------
        if plot_ss and ss_df is not None and ss_column_name is not None:
            mask_s = ss_df[ss_column_name].between(10 ** min_exp, 10 ** max_exp)
            counts_s, _ = np.histogram(ss_df.loc[mask_s, ss_column_name], bins=bins, density=True)
            # area_s = np.sum(counts_s * bin_widths)
            # print(f"[{self.filename}] steady-state PDF area = {area_s:.6f}")

        # ---------- ⑤ 绘图 ----------
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(f"Log({column_name})")
        plt.ylabel("Log(Probability Density)")
        plt.title(f"Log-Log Probability Density Function of {column_name}")
        plt.grid(True, which="both", ls="--")

        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        plt.scatter(bin_centers[counts_m > 0], counts_m[counts_m > 0],
                    color="#DC143C", s=35, marker="o", label="Model Result")

        if plot_ss and ss_df is not None and ss_column_name is not None:
            plt.scatter(bin_centers[counts_s > 0], counts_s[counts_s > 0],
                        color="#1E90FF", s=25, marker="D", label="Empirical Data")

            if plot_type == "weight":
                log_x = np.log(bin_centers[counts_s > 0])
                log_y = np.log(counts_s[counts_s > 0])
                slope, intercept, _, _, std_err = stats.linregress(log_x, log_y)
                plt.plot(bin_centers, np.exp(intercept) * bin_centers ** slope,
                         color="black", linestyle="--", linewidth=0.5,
                         label=f"SS Fit: k={slope:.2e} ± {std_err:.2e}, b={intercept:.2e}")

        exponents = np.arange(math.floor(min_exp), math.ceil(max_exp) + 1, dtype=float)
        plt.xticks(10.0 ** exponents, [f"$10^{{{int(e)}}}$" for e in exponents])

        plt.legend()
        plt.savefig(os.path.join(path, f"{param_name}+{self.filename}.png"))
        plt.close()
