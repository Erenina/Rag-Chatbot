"""RAG değerlendirme (evaluation) paketi.

Bu paket, RAG hattının ne kadar iyi çalıştığını ölçer:
  - retrieval doğru parçayı getiriyor mu?  (metrics.py — deterministik)
  - üretilen cevap sadık ve doğru mu?       (judge.py — LLM-as-a-judge)
Koşucu: run_eval.py  ->  python -m evals.run_eval
"""
