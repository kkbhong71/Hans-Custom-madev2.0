"""
=============================================================================
Han's Custom Lotto v2.0 - Web Application
Brother Lotto System v2.0 COVERAGE OPTIMIZED (Flask Web Version)
=============================================================================
"""

from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import random
import os
from collections import Counter
from itertools import combinations

app = Flask(__name__)

# =============================================================================
# [Part 1] 통계 유틸리티
# =============================================================================
class StatUtils:
    @staticmethod
    def calculate_ac(numbers):
        diffs = set()
        nums = sorted(numbers)
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                diffs.add(abs(nums[j] - nums[i]))
        return len(diffs) - (len(nums) - 1)

    @staticmethod
    def get_odd_count(numbers):
        return sum(1 for n in numbers if n % 2 != 0)

    @staticmethod
    def get_high_count(numbers):
        return sum(1 for n in numbers if n >= 23)

    @staticmethod
    def get_section_distribution(numbers):
        sections = [0] * 5
        for n in numbers:
            idx = min((n - 1) // 10, 4)
            sections[idx] += 1
        return sections

    @staticmethod
    def get_last_digit_sum(numbers):
        return sum(n % 10 for n in numbers)

    @staticmethod
    def has_triple_consecutive(numbers):
        nums = sorted(numbers)
        for i in range(len(nums) - 2):
            if nums[i + 1] == nums[i] + 1 and nums[i + 2] == nums[i] + 2:
                return True
        return False

    @staticmethod
    def get_consecutive_count(numbers):
        nums = sorted(numbers)
        return sum(1 for i in range(len(nums) - 1) if nums[i + 1] - nums[i] == 1)

    @staticmethod
    def count_zones(numbers):
        return len(set(min((n - 1) // 10, 4) for n in numbers))

    @staticmethod
    def get_ball_color(number):
        if 1 <= number <= 10: return '#FBC400'
        elif 11 <= number <= 20: return '#69C8F2'
        elif 21 <= number <= 30: return '#FF7272'
        elif 31 <= number <= 40: return '#AAAAAA'
        else: return '#B0D840'


# =============================================================================
# [Part 2] PoolAnalyzer
# =============================================================================
class PoolAnalyzer:
    def __init__(self, df, num_cols):
        self.df = df
        self.num_cols = num_cols

    def analyze(self, window):
        subset = self.df.head(window)
        all_nums = subset[self.num_cols].values.flatten()
        counts = Counter(all_nums)

        expected = window * 6 / 45
        p = 6 / 45
        std = np.sqrt(window * p * (1 - p))

        hot_pool, warm_pool, cold_pool = [], [], []
        hot_weights, warm_weights = [], []

        for num in range(1, 46):
            freq = counts.get(num, 0)
            z = (freq - expected) / std if std > 0 else 0

            if z > 0.5:
                hot_pool.append(num)
                hot_weights.append(freq)
            elif z >= -0.5:
                warm_pool.append(num)
                warm_weights.append(max(freq, 0.5))
            else:
                cold_pool.append(num)

        return {
            'hot_pool': hot_pool, 'hot_weights': hot_weights,
            'warm_pool': warm_pool, 'warm_weights': warm_weights,
            'cold_pool': cold_pool, 'expected': expected,
            'std': std, 'counts': counts, 'window': window
        }


# =============================================================================
# [Part 3] PopularityAdjuster
# =============================================================================
class PopularityAdjuster:
    BIRTHDAY_RANGE = set(range(1, 32))
    LUCKY_SEVENS = {7, 14, 21, 28, 35, 42}
    ROUND_NUMBERS = {10, 20, 30, 40}
    HIGH_RANGE = set(range(32, 46))

    @staticmethod
    def adjust(weights_dict):
        adjusted = {}
        for num, w in weights_dict.items():
            factor = 1.0
            if num in PopularityAdjuster.BIRTHDAY_RANGE: factor *= 0.93
            if num in PopularityAdjuster.LUCKY_SEVENS: factor *= 0.88
            if num in PopularityAdjuster.ROUND_NUMBERS: factor *= 0.93
            if num in PopularityAdjuster.HIGH_RANGE: factor *= 1.12
            adjusted[num] = w * factor
        return adjusted


# =============================================================================
# [Part 4] AdaptiveFilter
# =============================================================================
class AdaptiveFilter:
    def __init__(self, df, num_cols):
        nums_array = df[num_cols].values
        sums = nums_array.sum(axis=1)
        self.sum_min = int(np.percentile(sums, 2.5))
        self.sum_max = int(np.percentile(sums, 97.5))

        ac_values = [StatUtils.calculate_ac(sorted(row.tolist())) for row in nums_array]
        self.ac_min = int(np.percentile(ac_values, 5))

        odd_counts = [StatUtils.get_odd_count(row) for row in nums_array]
        self.allowed_odd = list(set(odd_counts))

        consec = [StatUtils.get_consecutive_count(row) for row in nums_array]
        self.max_consecutive = int(np.percentile(consec, 95))

        ld_sums = [StatUtils.get_last_digit_sum(row) for row in nums_array]
        self.ld_min = int(np.percentile(ld_sums, 2.5))
        self.ld_max = int(np.percentile(ld_sums, 97.5))

    def check(self, numbers, strict=True):
        nums = sorted(numbers)
        s = sum(nums)
        if strict:
            if not (self.sum_min <= s <= self.sum_max): return False
            if StatUtils.calculate_ac(nums) < self.ac_min: return False
            if StatUtils.get_odd_count(nums) not in self.allowed_odd: return False
            if StatUtils.get_consecutive_count(nums) > self.max_consecutive: return False
            ld = StatUtils.get_last_digit_sum(nums)
            if not (self.ld_min <= ld <= self.ld_max): return False
            if StatUtils.count_zones(nums) < 3: return False
            last_digits = [n % 10 for n in nums]
            if max(Counter(last_digits).values()) >= 3: return False
        else:
            if not (self.sum_min - 15 <= s <= self.sum_max + 15): return False
            if StatUtils.has_triple_consecutive(nums): return False
        return True


# =============================================================================
# [Part 5] NumberGenerator
# =============================================================================
class NumberGenerator:
    def __init__(self, adaptive_filter):
        self.filter = adaptive_filter

    def _weighted_sample(self, pool, weights, k=6):
        pool = list(pool)
        weights = np.array(weights, dtype=float)
        weights = np.maximum(weights, 0.01)
        weights = weights / weights.sum()
        if len(pool) < k:
            return sorted(pool)
        try:
            selected = np.random.choice(pool, size=k, replace=False, p=weights)
            return sorted([int(n) for n in selected])
        except ValueError:
            return sorted(random.sample(pool, min(k, len(pool))))

    def _uniform_sample(self, pool, k=6):
        return sorted(random.sample(list(pool), min(k, len(pool))))

    def algo_A_random(self, hot_pool, **kwargs):
        return self._uniform_sample(hot_pool)

    def algo_B_weighted(self, hot_pool, hot_weights, **kwargs):
        for _ in range(300):
            cand = self._weighted_sample(hot_pool, hot_weights)
            if self.filter.check(cand, strict=False):
                return cand
        return self._weighted_sample(hot_pool, hot_weights)

    def algo_C_balance(self, hot_pool, hot_weights, **kwargs):
        for _ in range(500):
            cand = self._weighted_sample(hot_pool, hot_weights)
            odd = StatUtils.get_odd_count(cand)
            high = StatUtils.get_high_count(cand)
            if 2 <= odd <= 4 and 2 <= high <= 4:
                return cand
        for _ in range(200):
            cand = self._weighted_sample(hot_pool, hot_weights)
            if 2 <= StatUtils.get_odd_count(cand) <= 4:
                return cand
        return self._weighted_sample(hot_pool, hot_weights)

    def algo_D_sum_range(self, hot_pool, hot_weights, **kwargs):
        for _ in range(500):
            cand = self._weighted_sample(hot_pool, hot_weights)
            s = sum(cand)
            if self.filter.sum_min <= s <= self.filter.sum_max:
                return cand
        return self._weighted_sample(hot_pool, hot_weights)

    def algo_E_pattern(self, hot_pool, hot_weights, **kwargs):
        for _ in range(500):
            cand = self._weighted_sample(hot_pool, hot_weights)
            sec = StatUtils.get_section_distribution(cand)
            if max(sec) >= 4: continue
            if StatUtils.has_triple_consecutive(cand): continue
            if StatUtils.count_zones(cand) < 3: continue
            if StatUtils.calculate_ac(cand) < self.filter.ac_min: continue
            return cand
        for _ in range(200):
            cand = self._weighted_sample(hot_pool, hot_weights)
            if not StatUtils.has_triple_consecutive(cand):
                return cand
        return self._weighted_sample(hot_pool, hot_weights)

    def algo_F_precision(self, hot_pool, hot_weights, **kwargs):
        for _ in range(3000):
            cand = self._weighted_sample(hot_pool, hot_weights)
            if self.filter.check(cand, strict=True):
                return cand
        for _ in range(1000):
            cand = self._weighted_sample(hot_pool, hot_weights)
            if self.filter.check(cand, strict=False):
                return cand
        return self._weighted_sample(hot_pool, hot_weights)

    def algo_G_hybrid(self, hot_pool, cold_pool, hot_weights, **kwargs):
        warm_pool = kwargs.get('warm_pool', [])
        cold_source = cold_pool + warm_pool if warm_pool else cold_pool
        if len(cold_source) < 1:
            return self.algo_F_precision(hot_pool, hot_weights)
        for _ in range(2000):
            mix = random.choice([(4, 2), (5, 1), (3, 3)])
            n_hot, n_cold = mix
            n_hot = min(n_hot, len(hot_pool))
            n_cold = min(n_cold, len(cold_source))
            if n_hot + n_cold < 6:
                n_hot = min(6, len(hot_pool))
                n_cold = 0
            try:
                h_w = np.array(hot_weights[:len(hot_pool)], dtype=float)
                h_w = h_w / h_w.sum()
                hot_picks = np.random.choice(hot_pool, size=n_hot, replace=False, p=h_w).tolist()
                cold_picks = random.sample(cold_source, n_cold)
            except (ValueError, IndexError):
                continue
            cand = sorted(set(hot_picks + cold_picks))
            if len(cand) != 6: continue
            if self.filter.check(cand, strict=False):
                return cand
        return sorted(random.sample(list(set(hot_pool + cold_source)), min(6, len(set(hot_pool + cold_source)))))

    def generate(self, algo_type, **kwargs):
        algo_map = {
            'A': self.algo_A_random, 'B': self.algo_B_weighted,
            'C': self.algo_C_balance, 'D': self.algo_D_sum_range,
            'E': self.algo_E_pattern, 'F': self.algo_F_precision,
            'G': self.algo_G_hybrid,
        }
        func = algo_map.get(algo_type)
        if func is None:
            return self._uniform_sample(kwargs.get('hot_pool', list(range(1, 46))))
        return func(**kwargs)


# =============================================================================
# [Part 6] CoverageOptimizer
# =============================================================================
class CoverageOptimizer:
    MAX_OVERLAP = 3

    @staticmethod
    def optimize(candidate_pool, target_count):
        if len(candidate_pool) <= target_count:
            return candidate_pool

        by_window = {}
        for item in candidate_pool:
            w = item[3]
            if w not in by_window:
                by_window[w] = []
            by_window[w].append(item)

        n_windows = len(by_window)
        if n_windows == 0:
            return candidate_pool[:target_count]

        per_window = target_count // n_windows
        remainder = target_count - per_window * n_windows
        sorted_windows = sorted(by_window.keys())
        priority_order = ['F', 'B', 'E', 'C', 'D', 'G', 'A']

        selected = []
        covered = set()

        for wi, w in enumerate(sorted_windows):
            pool = by_window[w]
            quota = per_window + (1 if wi < remainder else 0)

            if len(pool) <= quota:
                selected.extend(pool)
                for item in pool:
                    covered.update(item[2])
                continue

            sorted_pool = sorted(
                pool,
                key=lambda x: priority_order.index(x[0]) if x[0] in priority_order else 99
            )

            window_selected = [sorted_pool[0]]
            covered.update(sorted_pool[0][2])

            for _ in range(quota - 1):
                best = None
                best_score = -1

                for item in pool:
                    if item in window_selected: continue
                    numbers = set(item[2])
                    all_selected = selected + window_selected
                    if all_selected:
                        max_overlap = max(len(numbers & set(s[2])) for s in all_selected)
                    else:
                        max_overlap = 0
                    if max_overlap > CoverageOptimizer.MAX_OVERLAP: continue
                    new_nums = len(numbers - covered)
                    score = new_nums * 2 + (6 - max_overlap)
                    if score > best_score:
                        best_score = score
                        best = item

                if best is not None:
                    window_selected.append(best)
                    covered.update(best[2])
                else:
                    for item in pool:
                        if item not in window_selected:
                            window_selected.append(item)
                            covered.update(item[2])
                            break

            selected.extend(window_selected)

        return selected


# =============================================================================
# [Part 7] BrotherLottoSystem
# =============================================================================
class BrotherLottoSystem:
    ALGO_NAMES = [
        ('A', '순수랜덤'), ('B', '빈도가중'), ('C', '밸런스'),
        ('D', '합계구간'), ('E', '패턴분산'), ('F', '정밀필터'),
        ('G', '하이브리드'),
    ]

    ALGO_DESCRIPTIONS = {
        'A': '순수 랜덤 — Hot Pool 내 균등 추출',
        'B': '빈도 가중 — 출현 빈도 기반 가중 샘플링',
        'C': '밸런스 — 홀짝/고저 균형 최적화',
        'D': '합계 구간 — 데이터 기반 합계 범위 필터',
        'E': '패턴 분산 — 구간분포 + AC값 최적화',
        'F': '정밀 필터 — 전체 통계 필터 통과 (최고 품질)',
        'G': '하이브리드 — Hot/Cold 혼합 과적합 방지',
    }

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.num_cols = [f'num{i}' for i in range(1, 7)]
        self.filter = None
        self.pool_analyzer = None
        self.generator = None
        self.popularity = PopularityAdjuster()

    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        cols = list(self.df.columns)
        if len(cols) >= 7 and 'num1' not in cols:
            self.df.columns = ['round', 'date', 'num1', 'num2', 'num3',
                               'num4', 'num5', 'num6', 'bonus'][:len(cols)]
        for col in self.num_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        self.df = self.df.dropna(subset=self.num_cols)
        if 'round' in self.df.columns:
            self.df = self.df.sort_values(by='round', ascending=False).reset_index(drop=True)
        self.filter = AdaptiveFilter(self.df, self.num_cols)
        self.pool_analyzer = PoolAnalyzer(self.df, self.num_cols)
        self.generator = NumberGenerator(self.filter)
        return {
            'total_rounds': len(self.df),
            'latest_round': int(self.df.iloc[0]['round']) if 'round' in self.df.columns else len(self.df),
            'sum_range': f"{self.filter.sum_min}~{self.filter.sum_max}",
            'ac_min': self.filter.ac_min,
            'ld_range': f"{self.filter.ld_min}~{self.filter.ld_max}"
        }

    def get_pool_analysis(self, window):
        pool_data = self.pool_analyzer.analyze(window)
        hot_pool = pool_data['hot_pool']
        hot_weights = pool_data['hot_weights']
        warm_pool = pool_data['warm_pool']
        cold_pool = pool_data['cold_pool']

        # Build frequency data for chart
        freq_data = []
        for num in range(1, 46):
            freq = pool_data['counts'].get(num, 0)
            z = (freq - pool_data['expected']) / pool_data['std'] if pool_data['std'] > 0 else 0
            category = 'hot' if z > 0.5 else ('warm' if z >= -0.5 else 'cold')
            freq_data.append({
                'number': int(num),
                'frequency': int(freq),
                'z_score': round(z, 2),
                'category': category,
                'color': StatUtils.get_ball_color(num)
            })

        return {
            'hot_count': len(hot_pool),
            'warm_count': len(warm_pool),
            'cold_count': len(cold_pool),
            'expected': round(pool_data['expected'], 1),
            'std': round(pool_data['std'], 1),
            'freq_data': freq_data,
            'hot_numbers': hot_pool,
            'cold_numbers': cold_pool
        }

    def run(self, windows=None, sets_per_window=7, sets_per_window_map=None):
        """
        윈도우별 독립 실행: 각 구간에서 지정된 세트 수만큼 생성
        Args:
            windows: 분석 구간 리스트
            sets_per_window: 기본 세트 수 (sets_per_window_map이 없을 때 사용)
            sets_per_window_map: 구간별 개별 세트 수 {"30": 5, "50": 7, "100": 3}
        """
        if windows is None:
            windows = [30, 50, 100]

        windows_data = {}
        global_all_numbers = set()

        for w in windows:
            if w > len(self.df):
                continue

            # 구간별 세트 수 결정
            if sets_per_window_map and str(w) in sets_per_window_map:
                target_sets = int(sets_per_window_map[str(w)])
            else:
                target_sets = sets_per_window

            pool_data = self.pool_analyzer.analyze(w)
            hot_pool = pool_data['hot_pool']
            hot_weights = pool_data['hot_weights']
            warm_pool = pool_data['warm_pool']
            cold_pool = pool_data['cold_pool']

            if len(hot_pool) < 6:
                hot_pool = hot_pool + warm_pool
                hot_weights = hot_weights + pool_data['warm_weights']

            hot_pool, hot_weights = self._apply_popularity_adjustment(hot_pool, hot_weights)

            # 기본 7개 알고리즘으로 후보 생성
            candidates = []
            for code, name in self.ALGO_NAMES:
                nums = self.generator.generate(
                    code, hot_pool=hot_pool, hot_weights=hot_weights,
                    cold_pool=cold_pool, warm_pool=warm_pool
                )
                if nums and len(nums) == 6:
                    candidates.append((code, name, nums, w))

            # 추가 후보 생성 (target_sets > 7 일 때)
            extra_needed = max(0, target_sets * 2 - len(candidates))
            if extra_needed > 0:
                hp = pool_data['hot_pool'] + pool_data['warm_pool']
                hw = pool_data['hot_weights'] + pool_data['warm_weights']
                hp, hw = self._apply_popularity_adjustment(hp, hw)
                for _ in range(extra_needed):
                    algo = random.choice(['B', 'C', 'E', 'F'])
                    nums = self.generator.generate(
                        algo, hot_pool=hp, hot_weights=hw,
                        cold_pool=cold_pool, warm_pool=warm_pool
                    )
                    if nums and len(nums) == 6:
                        candidates.append((algo, dict(self.ALGO_NAMES).get(algo, ''), nums, w))

            # 구간별 커버리지 최적화
            optimized = CoverageOptimizer.optimize(candidates, target_sets)

            # 구간별 결과 빌드
            results = []
            window_numbers = set()
            for code, name, nums, window in optimized:
                sec = StatUtils.get_section_distribution(nums)
                ac = StatUtils.calculate_ac(nums)
                odd = StatUtils.get_odd_count(nums)
                s = sum(nums)
                window_numbers.update(nums)
                global_all_numbers.update(nums)

                results.append({
                    'algo_code': code,
                    'algo_name': name,
                    'algo_desc': self.ALGO_DESCRIPTIONS.get(code, ''),
                    'numbers': nums,
                    'colors': [StatUtils.get_ball_color(n) for n in nums],
                    'sum': s,
                    'odd_count': odd,
                    'ac': ac,
                    'sections': sec,
                    'window': window,
                })

            # 구간별 겹침 계산
            overlaps = []
            if len(optimized) >= 2:
                for i in range(len(optimized)):
                    for j in range(i + 1, len(optimized)):
                        overlap = len(set(optimized[i][2]) & set(optimized[j][2]))
                        overlaps.append(overlap)

            coverage = len(window_numbers)
            uncovered = sorted(set(range(1, 46)) - window_numbers)

            windows_data[str(w)] = {
                'results': results,
                'stats': {
                    'total_sets': len(results),
                    'coverage': coverage,
                    'coverage_pct': round(coverage / 45 * 100, 1),
                    'uncovered': uncovered,
                    'avg_overlap': round(np.mean(overlaps), 1) if overlaps else 0,
                    'max_overlap': max(overlaps) if overlaps else 0
                },
                'pool_info': {
                    'hot_count': len(pool_data['hot_pool']),
                    'warm_count': len(pool_data['warm_pool']),
                    'cold_count': len(pool_data['cold_pool']),
                    'expected': round(pool_data['expected'], 1),
                    'std': round(pool_data['std'], 1)
                }
            }

        # 글로벌 통계
        global_coverage = len(global_all_numbers)
        global_uncovered = sorted(set(range(1, 46)) - global_all_numbers)
        total_sets = sum(wd['stats']['total_sets'] for wd in windows_data.values())

        return {
            'windows_data': windows_data,
            'global_stats': {
                'total_sets': total_sets,
                'coverage': global_coverage,
                'coverage_pct': round(global_coverage / 45 * 100, 1),
                'uncovered': global_uncovered,
                'windows': list(windows_data.keys())
            }
        }

    def _apply_popularity_adjustment(self, pool, weights):
        w_dict = {num: w for num, w in zip(pool, weights)}
        adjusted = self.popularity.adjust(w_dict)
        adj_pool = list(adjusted.keys())
        adj_weights = list(adjusted.values())
        return adj_pool, adj_weights

    def get_recent_draws(self, n=10):
        recent = self.df.head(n)
        draws = []
        for _, row in recent.iterrows():
            nums = [int(row[col]) for col in self.num_cols]
            draws.append({
                'round': int(row.get('round', 0)),
                'date': str(row.get('draw date', row.get('date', ''))),
                'numbers': nums,
                'colors': [StatUtils.get_ball_color(n) for n in nums],
                'sum': sum(nums)
            })
        return draws


# =============================================================================
# Global system instance
# =============================================================================
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'new_1210.csv')
system = BrotherLottoSystem(csv_path)
system_info = system.load_data()


# =============================================================================
# Routes
# =============================================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/info')
def api_info():
    return jsonify(system_info)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json or {}
    windows = data.get('windows', [30, 50, 100])
    sets_per_window = data.get('sets_per_window', 7)
    sets_per_window_map = data.get('sets_per_window_map', None)
    results = system.run(windows=windows, sets_per_window=sets_per_window,
                         sets_per_window_map=sets_per_window_map)
    return jsonify(results)


@app.route('/api/pool/<int:window>')
def api_pool(window):
    analysis = system.get_pool_analysis(window)
    return jsonify(analysis)


@app.route('/api/recent')
def api_recent():
    n = request.args.get('n', 10, type=int)
    draws = system.get_recent_draws(n)
    return jsonify(draws)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
