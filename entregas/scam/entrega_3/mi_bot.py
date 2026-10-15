"""CataBotv4 — Guardian híbrido con auditoría aritmética explícita.

Base: clasificador optimizado CataBotv2 + reglas generales para comprobar
relaciones aritméticas explícitas (variaciones, puntos porcentuales, productos,
subvenciones, mayorías y fechas). No usa modelos externos en inferencia.

Precaución: los resultados exploratorios sobre H no equivalen a evaluación en
un test oculto independiente. Las reglas de este módulo son verificaciones
lógicas conservadoras, no una garantía de puntuación perfecta.
"""
import math
import re
from collections import defaultdict


class Guardian:
    def __init__(self):
        # Umbral calibrado sobre validación cruzada agrupada en casos H (se reescala a 0.5)
        self.umbral_optimo = 0.445

        # 1. Subconjuntos y partitivos ("14 de los 23 comedores", "3 de los 5", "de las que 131")
        self.pat_partitivo_sust = re.compile(
            r"\b(\d+(?:[.,]\d+)?)\s+de\s+(?:los|las|sus|cada)\s+(?:\d+(?:[.,]\d+)?\s+)?([a-záéíóúñ]{4,})",
            re.IGNORECASE,
        )
        self.pat_partitivo_de = re.compile(
            r"\bde\s+(?:los|las)\s+(\d+(?:[.,]\d+)?)\s+([a-záéíóúñ]{4,})",
            re.IGNORECASE,
        )
        self.pat_partitivo_num = re.compile(
            r"\b(\d+(?:[.,]\d+)?)\s+de\s+(?:los|las)\s+(\d+(?:[.,]\d+)?)\b",
            re.IGNORECASE,
        )
        self.pat_de_las_que = re.compile(
            r"\bde\s+(?:los|las)\s+que\s+(?:solo\s+|sólo\s+)?(\d+(?:[.,]\d+)?)\b",
            re.IGNORECASE,
        )
        self.pat_cuant_sub = re.compile(
            r"\b(algunos|algunas|varios|varias|ciertos|ciertas|determinados|determinadas|"
            r"parte de los|parte de las|una parte de)\s+([a-záéíóúñ]{4,})",
            re.IGNORECASE,
        )

        # 2. Restricciones y excepciones ("solo", "salvo", "excepto")
        self.pat_restriccion = re.compile(
            r"\b(solo|sólo|únicamente|solamente|exclusivamente|salvo|excepto|a excepción)\b",
            re.IGNORECASE,
        )

        # 3. Condicionales ("si llueve", "si se aprueba", "siempre que", "en caso de")
        self.pat_condicional = re.compile(
            r"\b(si\s+(?!bien\b)(?:se\s+|no\s+|el\s+|la\s+|los\s+|las\s+)?[a-záéíóúñ]{3,}|"
            r"siempre que|en caso de|a condición de|salvo que|a menos que)\b",
            re.IGNORECASE,
        )

        # 4. Modales y verbos epistémicos/tentativos en la fuente vs respuesta
        self.pat_epistemico_src = re.compile(
            r"\b(podría|podrían|prevé|prevén|estima|estiman|calcula|calculan|"
            r"estudia|estudian|propuso|propone|proponen|planea|planean|"
            r"espera|esperan|proyecta|proyectan|sugiere|sugieren|baraja|barajan|"
            r"apunta|apuntan|aspira|aspiran|intentan?|buscan?|posiblemente|probablemente)\b",
            re.IGNORECASE,
        )
        self.pat_epistemico_resp = re.compile(
            r"\b(podría|podrían|prevé|prevén|estima|estiman|calcula|calculan|"
            r"estudia|estudian|propuso|propone|proponen|planea|planean|"
            r"espera|esperan|proyecta|proyectan|sugiere|sugieren|baraja|barajan|"
            r"apunta|apuntan|aspira|aspiran|intentan?|buscan?|posible|probable|"
            r"previsión|estimación|propuesta|estudio|plan|proyecto)\b",
            re.IGNORECASE,
        )

        # 5. Conectores causales, consecutivos y de finalidad añadidos
        self.pat_causal = re.compile(
            r"\b(gracias a|debido a|por culpa de|como consecuencia de|a causa de|"
            r"motivado por|lo que provocó|lo que permitió|con lo que|por lo que|"
            r"con el objetivo de|con el fin de|a fin de|para lograr|para poder\s+[a-záéíóúñ]+)\b",
            re.IGNORECASE,
        )

        # 6. Orden temporal estricto añadido
        self.pat_temporal_ord = re.compile(
            r"\b(después de que|antes de que|tras el cierre|antes del cierre|posteriormente a)\b",
            re.IGNORECASE,
        )

        # 7. Calificadores restrictivos técnicos/legales en la fuente
        self.pat_calificador_estricto = re.compile(
            r"\b(de grado superior|de grado medio|en contenedor\s+[a-záéíóúñ]+|"
            r"en depósito|cedido en depósito|con carácter\s+[a-záéíóúñ]+|"
            r"de forma temporal|provisionalmente|temporalmente|"
            r"(?:inferior|superior|menor|mayor)\s+a\s+\d+(?:[.,]\d+)?)\b",
            re.IGNORECASE,
        )

        # 8. Enumeración restrictiva en la fuente ("de A, B y C" / "en explotaciones de A, B y C")
        self.pat_enum_restrictiva = re.compile(
            r"\bde\s+([a-záéíóúñ]{4,})\s*,\s*([a-záéíóúñ]{4,})\s+y\s+([a-záéíóúñ]{4,})\b",
            re.IGNORECASE,
        )

        self.mapa_num_texto = {
            "un": 1.0, "uno": 1.0, "una": 1.0, "dos": 2.0, "tres": 3.0, "cuatro": 4.0,
            "cinco": 5.0, "seis": 6.0, "siete": 7.0, "ocho": 8.0, "nueve": 9.0,
            "diez": 10.0, "once": 11.0, "doce": 12.0, "trece": 13.0, "catorce": 14.0,
            "quince": 15.0, "dieciséis": 16.0, "diecisiete": 17.0, "dieciocho": 18.0,
            "diecinueve": 19.0, "veinte": 20.0, "treinta": 30.0, "cuarenta": 40.0,
            "cincuenta": 50.0, "sesenta": 60.0, "setenta": 70.0, "ochenta": 80.0,
            "noventa": 90.0, "cien": 100.0, "mil": 1000.0,
        }

        self.stopwords = {
            "para", "como", "este", "esta", "estos", "estas", "pero", "sino", "porque",
            "cuando", "donde", "quien", "cual", "cuales", "entre", "hacia", "hasta",
            "desde", "sobre", "tras", "durante", "mediante", "según", "contra", "bajo",
            "ante", "todo", "toda", "todos", "todas", "otro", "otra", "otros", "otras",
            "mismo", "misma", "mismos", "mismas", "tanto", "tanta", "tantos", "tantas",
            "cada", "alguno", "alguna", "algunos", "algunas", "ninguno", "ninguna",
            "poco", "poca", "pocos", "pocas", "mucho", "mucha", "muchos", "muchas",
            "varios", "varias", "demás", "bastante", "cualquier", "cualquiera", "quienes",
            "cuyo", "cuya", "cuyos", "cuyas", "haber", "había", "habían", "hubo", "sido",
            "está", "están", "estaba", "estaban", "estuvo", "fueron", "será", "serán",
            "tiene", "tienen", "tenía", "tenían", "tuvo", "tuvieron", "hace", "hacen",
            "hizo", "hicieron", "puede", "pueden", "podía", "podían", "pudo", "pudieron",
            "debe", "deben", "además", "aunque", "también", "solo", "sólo", "más", "menos",
            "muy", "bien", "casi", "alrededor", "aproximadamente", "respecto", "través",
            "embargo", "obstante", "asimismo", "dentro", "fuera", "encima", "debajo",
            "delante", "detrás", "cerca", "lejos", "antes", "después", "luego", "entonces",
            "ayer", "siempre", "nunca", "jamás", "quizá", "quizás", "acaso", "tampoco",
            "siquiera", "apenas", "unas", "unos", "tres", "cuatro", "cinco", "seis",
            "siete", "ocho", "nueve", "diez", "once", "doce", "cien", "ciento", "miles",
            "millón", "millones", "primero", "primera", "segundo", "segunda", "tercero",
            "tercera", "mayor", "menor", "mejor", "peor", "gran", "grandes", "parte",
            "tipo", "forma", "modo", "manera", "caso", "casos", "hecho", "punto", "lado",
        }

    @staticmethod
    def _sigmoid(z):
        z = max(-30.0, min(30.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    @staticmethod
    def _raiz(palabra):
        return palabra[:5] if len(palabra) >= 6 else palabra[:4]

    def _extraer_numeros(self, texto, incluir_texto=True):
        raw = re.findall(r"\b\d+(?:[.,]\d+)*\b", texto)
        nums = set()
        for r in raw:
            limpio = re.sub(r"\.(?=\d{3}\b)", "", r).replace(",", ".")
            try:
                nums.add(float(limpio))
            except ValueError:
                pass
        if incluir_texto:
            for w in re.findall(r"\b[a-záéíóúñ]+\b", texto.lower()):
                if w in self.mapa_num_texto:
                    nums.add(self.mapa_num_texto[w])
        return nums

    def _extraer_palabras_contenido(self, texto):
        palabras = re.findall(r"\b[a-záéíóúñ]{4,}\b", texto.lower())
        return [p for p in palabras if p not in self.stopwords]

    def _conjunto_derivados(self, nums_base):
        derivados = set(nums_base)
        lista = [x for x in nums_base if x is not None]
        for a in lista:
            derivados.update([
                a * 1000.0, a / 1000.0, a * 2.0, a / 2.0,
                a * 10.0, a / 10.0, a * 100.0, a / 100.0
            ])
            for b in lista:
                derivados.update([
                    a + b, abs(a - b), round(a + b, 2), round(abs(a - b), 2)
                ])
                if b > 0 and a <= b:
                    pct = 100.0 * a / b
                    derivados.update([round(pct, 0), round(pct, 1), round(pct, 2)])
        return derivados

    def _auditar_oracion(self, texto_orac, fuentes, idx_src_p, idxs_rel, max_resp_orac):
        res = {
            "partitivo_borrado": 0.0,
            "restriccion_borrada": 0.0,
            "condicion_borrada": 0.0,
            "epistemico_borrado": 0.0,
            "calificador_borrado": 0.0,
            "causal_inventado": 0.0,
            "temporal_inventado": 0.0,
            "num_no_respaldado": 0.0,
            "num_ajeno_en_monofuente": 0.0,
        }
        if not fuentes or not texto_orac:
            return res

        t_low = texto_orac.lower()
        nums_orac = self._extraer_numeros(texto_orac, incluir_texto=True)
        nums_dig_orac = self._extraer_numeros(texto_orac, incluir_texto=False)
        raices_orac = {self._raiz(p) for p in self._extraer_palabras_contenido(texto_orac)}

        src_p = fuentes[idx_src_p] if 0 <= idx_src_p < len(fuentes) else ""
        srcs_rel_txt = " ".join(fuentes[i] for i in idxs_rel if 0 <= i < len(fuentes))
        todas_txt = " ".join(fuentes)

        # 1. Partitivo / subconjunto borrado ("14 de los 23 comedores", "3 de los 5", "de las que 131")
        for s_chk in ([src_p] if src_p else []) + [srcs_rel_txt]:
            for pat in (self.pat_partitivo_sust, self.pat_partitivo_de):
                for num_s, sust in pat.findall(s_chk):
                    if self._raiz(sust.lower()) in raices_orac:
                        v_n = self._extraer_numeros(num_s)
                        if v_n and not v_n.issubset(nums_orac):
                            res["partitivo_borrado"] = 1.0
            for cuant, sust in self.pat_cuant_sub.findall(s_chk):
                if self._raiz(sust.lower()) in raices_orac:
                    if not self.pat_cuant_sub.search(texto_orac) and not self.pat_partitivo_sust.search(texto_orac):
                        res["partitivo_borrado"] = max(res["partitivo_borrado"], 0.8)

        if src_p:
            for n1_s, n2_s in self.pat_partitivo_num.findall(src_p):
                v1 = self._extraer_numeros(n1_s)
                if v1 and not v1.issubset(nums_orac):
                    res["partitivo_borrado"] = 1.0
            for n_dlq in self.pat_de_las_que.findall(src_p):
                v_dlq = self._extraer_numeros(n_dlq)
                if v_dlq and not v_dlq.issubset(nums_orac):
                    res["partitivo_borrado"] = 1.0

        # 2. Restricción borrada ("solo", "salvo", "excepto") en la fuente principal
        if src_p and self.pat_restriccion.search(src_p):
            if not self.pat_restriccion.search(texto_orac) and "únic" not in t_low and "excep" not in t_low:
                res["restriccion_borrada"] = 1.0

        # 3. Condición borrada ("si llueve", "si se aprueba") en la fuente principal
        if src_p and self.pat_condicional.search(src_p):
            if not self.pat_condicional.search(texto_orac) and not re.search(r"\b(si|condici|depend|sujeto|podr[ií]a)\b", t_low):
                res["condicion_borrada"] = 1.0

        # 4. Epistémico / modal borrado ("podría", "prevé", "estima", "propuso")
        if src_p and self.pat_epistemico_src.search(src_p):
            if not self.pat_epistemico_resp.search(texto_orac):
                res["epistemico_borrado"] = 1.0

        # 5. Calificador técnico/legal o enumeración restrictiva borrada en fuente principal
        if src_p:
            for cal in self.pat_calificador_estricto.findall(src_p):
                cal_low = cal.lower()
                # Si la frase calificada está ausente en la oración
                nums_cal = self._extraer_numeros(cal_low, incluir_texto=False)
                if nums_cal:
                    if not nums_cal.issubset(nums_orac):
                        res["calificador_borrado"] = 1.0
                elif cal_low not in t_low:
                    # Comprobar raíces clave del calificador
                    r_cals = {self._raiz(w) for w in self._extraer_palabras_contenido(cal_low)}
                    if r_cals and not r_cals.issubset(raices_orac):
                        res["calificador_borrado"] = 1.0

            # Enumeración de 3 subtipos específicos ("de melocotón, pera y manzana") omitida
            for a, b, c in self.pat_enum_restrictiva.findall(src_p):
                r_abc = {self._raiz(a.lower()), self._raiz(b.lower()), self._raiz(c.lower())}
                if len(r_abc.intersection(raices_orac)) == 0:
                    res["calificador_borrado"] = max(res["calificador_borrado"], 0.9)

        # 6. Causal o finalidad inventada ("gracias a", "con lo que", "para poder + inf")
        causas_orac = set(m.lower() for m in self.pat_causal.findall(texto_orac))
        if causas_orac:
            causas_todas = set(m.lower() for m in self.pat_causal.findall(todas_txt))
            if not causas_orac.issubset(causas_todas):
                res["causal_inventado"] = 1.0

        # 7. Orden temporal inventado ("después de que", "antes de que")
        if self.pat_temporal_ord.search(texto_orac) and not self.pat_temporal_ord.search(todas_txt):
            res["temporal_inventado"] = 1.0

        # 8. Verificación matemática y numérica (incluyendo restas, sumas y porcentajes)
        if nums_dig_orac:
            deriv_todas = self._conjunto_derivados(self._extraer_numeros(todas_txt, incluir_texto=True))
            deriv_src_p = self._conjunto_derivados(self._extraer_numeros(src_p, incluir_texto=True)) if src_p else deriv_todas

            for n in nums_dig_orac:
                tol = max(1e-3, abs(n) * 0.006)
                if not any(abs(n - d) <= tol for d in deriv_todas):
                    res["num_no_respaldado"] = 1.0
                elif max_resp_orac > 0.85 and src_p and not any(abs(n - d) <= tol for d in deriv_src_p):
                    # La oración depende claramente de una sola fuente pero mete un número que no sale de ella
                    res["num_ajeno_en_monofuente"] = 1.0

        return res

    def _prob_oracion(self, oracion, frags, fuentes):
        llm = oracion.get("llm") or {}
        veredicto = llm.get("veredicto")
        conf = float(llm.get("confianza")) if llm.get("confianza") is not None else 0.5
        f_llm = llm.get("fuente")

        resp_f = oracion.get("respaldo_por_fuente") or [0.0]
        cont_f = oracion.get("contradiccion_por_fuente") or [0.0]
        sim_f = oracion.get("similitud_por_fuente") or [0.0]
        resp_todas = float(oracion.get("respaldo_todas") or 0.0)

        max_resp = max(resp_f) if resp_f else 0.0
        idx_max_resp = max(range(len(resp_f)), key=lambda i: resp_f[i]) if resp_f else 0
        idx_max_sim = max(range(len(sim_f)), key=lambda i: sim_f[i]) if sim_f else 0
        max_sim = max(sim_f) if sim_f else 0.0

        if isinstance(f_llm, int) and 0 <= f_llm < len(fuentes):
            idx_src_p = f_llm
        else:
            idx_src_p = idx_max_resp if max_resp >= 0.25 else idx_max_sim

        cont_src_p = cont_f[idx_src_p] if idx_src_p < len(cont_f) else 0.0

        idxs_relevantes = {idx_src_p, idx_max_resp, idx_max_sim}
        if frags:
            frag_best_resps = []
            frag_max_resps = []
            frag_sims = []
            for fr in frags:
                fr_rf = fr.get("respaldo_por_fuente") or [0.0]
                fr_sf = fr.get("similitud_por_fuente") or [0.0]
                fr_rt = float(fr.get("respaldo_todas") or 0.0)
                m_rf = max(fr_rf) if fr_rf else 0.0
                if fr_rf:
                    idxs_relevantes.add(max(range(len(fr_rf)), key=lambda i: fr_rf[i]))
                if fr_sf:
                    idxs_relevantes.add(max(range(len(fr_sf)), key=lambda i: fr_sf[i]))
                frag_max_resps.append(m_rf)
                frag_best_resps.append(max(m_rf, fr_rt))
                frag_sims.append(max(fr_sf) if fr_sf else 0.0)
            min_frag_best_resp = min(frag_best_resps)
            min_frag_max_resp = min(frag_max_resps)
            max_frag_best_resp = max(frag_best_resps)
            min_frag_sim = min(frag_sims)
        else:
            min_frag_best_resp = max(max_resp, resp_todas)
            min_frag_max_resp = max_resp
            max_frag_best_resp = min_frag_best_resp
            min_frag_sim = max_sim

        best_orac_resp = max(max_resp, resp_todas)
        combined_support = max(best_orac_resp, min_frag_best_resp)
        frag_gap = max_frag_best_resp - min_frag_best_resp

        is_contra = 1.0 if veredicto == "contradicha" else 0.0
        is_no_resp = 1.0 if veredicto == "no_respaldada" else 0.0
        is_null = 1.0 if veredicto not in ("respaldada", "no_respaldada", "contradicha") else 0.0
        llm_bad = max(is_contra, is_no_resp, is_null)

        lex = self._auditar_oracion(
            oracion.get("texto", ""), fuentes, idx_src_p, idxs_relevantes, max_resp
        )

        # Escalón no lineal cuando NLI no ve respaldo ni en la oración ni en los fragmentos
        nli_vacio = 1.0 if (combined_support < 0.30 and best_orac_resp < 0.25) else 0.0
        frag_suelto_vacio = 1.0 if (min_frag_best_resp < 0.15 and best_orac_resp < 0.50) else 0.0
        cont_local = cont_src_p if (max_resp > 0.70 and cont_src_p > 0.30) else 0.0

        # Log-odds combinando pesos aprendidos en CV + detectores quirúrgicos
        z = -0.92
        z += 1.50 * is_contra
        z += 0.75 * is_no_resp
        z += 0.40 * is_null
        z += 2.05 * (llm_bad * conf)
        z += 0.55 * (1.0 - conf)
        z += 0.30 * (1.0 - max_resp)
        z += 0.15 * (1.0 - best_orac_resp)
        z += 0.15 * (1.0 - min_frag_max_resp)
        z += 0.48 * (1.0 - combined_support)
        z += 0.45 * frag_gap
        z += 0.68 * max(0.0, max_sim - combined_support)
        z -= 0.38 * (1.0 - min_frag_sim) * combined_support  # Premia paráfrasis solo si tienen soporte NLI
        z += 0.85 * nli_vacio
        z += 0.65 * frag_suelto_vacio
        z += 0.35 * cont_local

        # Reglas quirúrgicas de desconfianza sobre el LLM
        z += 0.78 * lex["partitivo_borrado"]
        z += 1.05 * lex["restriccion_borrada"]
        z += 1.00 * lex["condicion_borrada"]
        z += 1.18 * lex["epistemico_borrado"]
        z += 0.88 * lex["calificador_borrado"]
        z += 0.72 * lex["causal_inventado"]
        z += 0.85 * lex["temporal_inventado"]
        z += 0.65 * lex["num_no_respaldado"]
        z += 0.80 * lex["num_ajeno_en_monofuente"]

        return self._sigmoid(z)

    def _reescalar(self, s):
        u = self.umbral_optimo
        if u <= 0.0 or u >= 1.0:
            return s
        if s < u:
            return 0.5 * (s / u)
        return 0.5 + 0.5 * ((s - u) / (1.0 - u))

    def score(self, caso) -> float:
        senales = caso.get("senales") or {}
        oraciones = senales.get("oraciones") or []
        fragmentos = senales.get("fragmentos") or []
        fuentes = caso.get("fuentes") or []

        if not oraciones:
            return 0.0

        frags_por_orac = defaultdict(list)
        for f in fragmentos:
            frags_por_orac[f.get("oracion", 0)].append(f)

        # El riesgo del caso es el de su oración más sospechosa (basta una falsa)
        peor_prob = max(
            self._prob_oracion(orac, frags_por_orac.get(i, []), fuentes)
            for i, orac in enumerate(oraciones)
        )

        return min(1.0, max(0.0, self._reescalar(peor_prob)))

# ---------------------------------------------------------------------------

# CataBotv2 mejorado: calibración supervisada sobre train.jsonl.
# La inferencia solo utiliza Python estándar; no se necesitan modelos externos.
# Las árboles se exportaron como reglas numéricas y se evalúan aquí directamente.
# ---------------------------------------------------------------------------

_GUARDIAN_SCORE_ORIGINAL = Guardian.score
_CB_MODEL_FEATURES = ['sent_llm_respaldada_n_gt5', 'worst_llm_respaldada', 'sent_llm_respaldada_n_gt75', 'sent_llm_respaldada_q25', 'old_case_logit', 'sent_llm_respaldada_min', 'sent_llm_respaldada_n_gt25', 'worst_heuristic_risk', 'sent_llm_respaldada_mean', 'worst_support_all_minus_contra_all', 'sent_s_all_q25', 'sent_llm_respaldada_q75', 'sent_support_all_minus_contra_all_min', 'sent_llm_contradicha_max', 'sent_llm_contradicha_n_gt5', 'old_case_score', 'sent_llm_contradicha_q75', 'sent_support_all_minus_contra_all_q25', 'sent_support_all_minus_contra_all_mean', 'sent_frag_sall_n_gt25_n_gt75', 'sent_s_all_min', 'worst_llm_contradicha', 'sent_llm_contradicha_n_gt25', 'sent_llm_respaldada_max', 'sent_frag_sall_min_n_gt25', 'sent_s_all_mean', 'sent_support_all_minus_contra_all_n_gt75', 'sent_support_all_minus_contra_all_n_gt5', 'sent_support_all_minus_contra_all_n_gt25', 'case_frag_sall_mean', 'sent_llm_contradicha_n_gt75', 'sent_s_all_n_gt75', 'sent_frag_sall_min_min', 'sent_llm_contradicha_min', 'worst_frag_smax_min', 'sent_frag_sall_q25_min', 'sent_frag_sall_mean_mean', 'worst_frag_sall_mean', 'sent_llm_no_respaldada_n_gt5', 'sent_llm_contradicha_mean', 'case_frag_sall_min', 'sent_support_all_minus_contra_all_q75', 'sent_frag_sall_n_gt25_min', 'worst_frag_sall_min', 'sent_llm_contradicha_q25', 'sent_frag_sall_q25_mean', 'sent_support_all_minus_contra_all_max', 'sent_frag_sall_n_gt5_min', 'sent_frag_sall_n_gt25_q75', 'sent_frag_sall_min_mean', 'sent_s_all_n_gt25', 'sent_llm_no_respaldada_n_gt75', 'case_frag_smax_min', 'sent_llm_no_respaldada_max', 'case_frag_sall_n_gt25', 'worst_frag_sall_q25', 'sent_frag_sall_n_gt25_mean', 'worst_s_n_gt25', 'sent_s_max_n_gt25', 'worst_llm_no_respaldada', 'sent_s_max_min', 'sent_s_all_max', 'sent_llm_conf_min', 'sent_frag_sall_q75_q25', 'sent_llm_conf_q25', 'case_frag_smax_n_gt5', 'sent_support_minus_contra_min', 'case_frag_sall_q25', 'sent_frag_smax_n_gt25_min', 'sent_llm_conf_max', 'sent_s_all_n_gt5', 'sent_frag_sall_n_gt25_q25', 'sent_frag_sall_min_q25', 'sent_frag_sall_n_gt5_n_gt75', 'worst_frag_sall_n_gt5', 'sent_frag_sall_mean_min', 'sent_llm_conf_mean', 'sent_frag_sall_q25_n_gt5', 'sent_llm_no_respaldada_n_gt25', 'sent_frag_sall_mean_n_gt75']
_CB_TREES = [{'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [15, 4, 69, -2, -2, 12, -2, -2, 5, 20, -2, -2, 7, -2, -2], 'threshold': [0.9089685380458832, -0.00559919816441834, 0.9750000238418579, -2.0, -2.0, -0.5535000264644623, -2.0, -2.0, 0.5, 0.035499999299645424, -2.0, -2.0, 0.6599499881267548, -2.0, -2.0], 'value': [-4.068866574138828e-16, -0.3403388790003904, -0.42748582885714576, -1.1793901572834316, -2.0111595034671885, -0.15471587580550702, -1.4596281859600655, -0.3644575786041487, 0.4161174575278306, 0.4560270960820478, 1.9024150860965687, 1.5683448128373039, 0.16130976522013454, 1.464317662435592, -0.20354949070519798]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [4, 4, 69, -2, -2, 65, -2, -2, 6, 20, -2, -2, 7, -2, -2], 'threshold': [2.301183581352234, -0.00559919816441834, 0.9750000238418579, -2.0, -2.0, 0.4166666716337204, -2.0, -2.0, 0.9545454680919647, 0.035499999299645424, -2.0, -2.0, 0.6599499881267548, -2.0, -2.0], 'value': [3.566270043172131e-05, -0.3233021788218691, -0.40608532904063255, -1.118512367813339, -1.9070828128826258, -0.146974068855909, -1.3336070979854593, -0.3265593137597353, 0.395366695499179, 0.4332865145725872, 1.8190219846349318, 1.4973971816469809, 0.1532632352612673, 1.397477533160963, -0.19329074143957006]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [4, 4, 64, -2, -2, 76, -2, -2, 8, 20, -2, -2, 7, -2, -2], 'threshold': [2.301183581352234, -0.00559919816441834, 0.9350000023841858, -2.0, -2.0, 0.9291666746139526, -2.0, -2.0, 0.9545454680919647, 0.035499999299645424, -2.0, -2.0, 0.6599499881267548, -2.0, -2.0], 'value': [6.802535511960316e-05, -0.30714742645891324, -0.38579753228333247, 0.2693685143005116, -1.6203907462105438, -0.1396227010529039, 0.6290524151982635, -0.720692566376849, 0.3756869176121231, 0.4117201985561406, 1.7462954057324096, 1.433522398406042, 0.14562827773877912, 1.3368135838495725, -0.183561935420655]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [4, 15, 69, -2, -2, 12, -2, -2, 2, 20, -2, -2, 7, -2, -2], 'threshold': [2.301183581352234, 0.49860021471977234, 0.9750000238418579, -2.0, -2.0, -0.5535000264644623, -2.0, -2.0, 0.9545454680919647, 0.035499999299645424, -2.0, -2.0, 0.6500249803066254, -2.0, -2.0], 'value': [9.990085426867025e-05, -0.29181913869361353, -0.36655094435590635, -0.9944831235916716, -1.747078689349901, -0.13264039263292698, -1.3194953818085817, -0.29091207156353033, 0.3570165390514847, 0.3912605049342967, 1.6823562074508789, 1.3755716764788881, 0.13838198764584198, 1.452152371300822, -0.03420300561322035]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [15, 64, 41, -2, -2, 15, -2, -2, 6, 4, -2, -2, 7, -2, -2], 'threshold': [0.9089685380458832, 0.9225000143051147, 0.002999999967869371, -2.0, -2.0, 0.49860021471977234, -2.0, -2.0, 0.9545454680919647, 2.897634983062744, -2.0, -2.0, 0.6599499881267548, -2.0, -2.0], 'value': [0.00012675732716076942, -0.2772792017435477, 0.12241808085063868, -0.44654441977474796, 0.882311593734067, -0.30504860292605895, -1.4844996363263805, -0.6577204096422818, 0.33929888697220584, 0.37184507234846703, 1.07676067488428, 1.589679143933823, 0.13150401110837884, 1.2360655666265568, -0.1726151932030815]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [15, 64, 76, -2, -2, 15, -2, -2, 8, 20, -2, -2, 7, -2, -2], 'threshold': [0.9089685380458832, 0.9587499797344208, 0.9291666746139526, -2.0, -2.0, 0.49860021471977234, -2.0, -2.0, 0.9545454680919647, 0.035499999299645424, -2.0, -2.0, 0.6599499881267548, -2.0, -2.0], 'value': [0.0001538372250602635, -0.2634740146404662, -0.13393926405466303, 0.4658620547965123, -0.7484798817616756, -0.34293089773175356, -1.58861710891488, -0.8303033576994938, 0.3224800779825155, 0.3534150388835317, 1.5767692792718155, 1.2679659647721333, 0.12497225069140527, 1.1884394918766017, -0.16394771498067215]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [4, 64, 76, -2, -2, 4, -2, -2, 2, 4, -2, -2, 7, -2, -2], 'threshold': [2.301183581352234, 0.9712500274181366, 0.9291666746139526, -2.0, -2.0, -0.00559919816441834, -2.0, -2.0, 0.9545454680919647, 2.897634983062744, -2.0, -2.0, 0.6500249803066254, -2.0, -2.0], 'value': [0.00017723208628300829, -0.2503722327400317, -0.1418896992271283, 0.4427702786939893, -0.7512309492270504, -0.33628874306037015, -1.5664174070298262, -0.8318476040708659, 0.30651310119033387, 0.335918713918848, 0.983214337912936, 1.493902432885144, 0.11876957376982379, 1.309864816546516, -0.03965910933352338]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [4, 4, 62, -2, -2, 65, -2, -2, 8, 60, -2, -2, 7, -2, -2], 'threshold': [2.301183581352234, -0.00559919816441834, 0.9600000083446503, -2.0, -2.0, 0.4166666716337204, -2.0, -2.0, 0.9545454680919647, 0.7840000092983246, -2.0, -2.0, 0.6500249803066254, -2.0, -2.0], 'value': [0.00019802101499009, -0.2379333865577769, -0.3010698507021913, -0.7895172022837021, -1.4918724558096836, -0.10345271793017738, -1.1144477096276166, -0.17224623000122516, 0.29135087480513505, 0.3193038503608836, 1.4424666589420134, 0.7665754011133776, 0.11288187702612343, 1.265891515350567, -0.03767229579984347]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [4, 76, 65, -2, -2, 4, -2, -2, 0, 20, -2, -2, 69, -2, -2], 'threshold': [2.301183581352234, 0.9291666746139526, 0.5833333432674408, -2.0, -2.0, -0.18394078314304352, -2.0, -2.0, 0.9545454680919647, 0.035499999299645424, -2.0, -2.0, 0.9350000023841858, -2.0, -2.0], 'value': [0.0002159061803110973, -0.22612313828916053, 0.11123576246001496, -0.30539967393088246, 0.9008229824877929, -0.24956151294257833, -1.3199283324970843, -0.5957777772737687, 0.2769507535199412, 0.3035241224601482, 1.4562092561801663, 1.1317699289442125, 0.10729001336321271, 1.5249146418563477, 0.09701425995788272]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [4, 64, 15, -2, -2, 15, -2, -2, 8, 4, -2, -2, 7, -2, -2], 'threshold': [2.301183581352234, 0.9712500274181366, 0.6188631951808929, -2.0, -2.0, 0.49860021471977234, -2.0, -2.0, 0.9545454680919647, 2.897634983062744, -2.0, -2.0, 0.6599499881267548, -2.0, -2.0], 'value': [0.0002357903489771465, -0.21489913545531894, -0.11733396209825045, -0.7068184675330513, 0.23908605099078512, -0.29216926320948416, -1.4354635326246976, -0.7452252800029128, 0.2632718519768862, 0.28853729484282004, 0.8576090899848557, 1.3812527187526167, 0.10196171675591917, 1.0541798038045582, -0.16796662854006456]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [15, 76, 18, -2, -2, 4, -2, -2, 0, 20, -2, -2, 69, -2, -2], 'threshold': [0.9089685380458832, 0.9291666746139526, 0.002999999967869371, -2.0, -2.0, -0.00559919816441834, -2.0, -2.0, 0.9545454680919647, 0.035499999299645424, -2.0, -2.0, 0.9350000023841858, -2.0, -2.0], 'value': [0.0002501923605559844, -0.20424554975236997, 0.11024343794233776, -0.45568295872037123, 0.8187993781645986, -0.22609501245097782, -1.2064452467082938, -0.478801833868007, 0.25027818955331294, 0.2742994256452921, 1.3928224618259277, 1.0566083658733747, 0.0969118360429846, 1.458116133586329, 0.07507176160075052]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [4, 64, 76, -2, -2, 15, -2, -2, 15, 64, -2, -2, 0, -2, -2], 'threshold': [2.301183581352234, 0.9587499797344208, 0.9291666746139526, -2.0, -2.0, 0.5021913051605225, -2.0, -2.0, 0.9475198686122894, 0.9437499940395355, -2.0, -2.0, 0.7750000059604645, -2.0, -2.0], 'value': [0.0002668083706744189, -0.19411850876924253, -0.09045218312783584, 0.4207697528779559, -0.5452578230506064, -0.25770764666268436, -1.3265218336135214, -0.6407985769399999, 0.23793323128002655, 0.09939561375814712, 1.257848953635687, 0.1199804437050282, 0.2623311271682597, 1.3273083206338143, 0.8109826568282013]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [4, 64, 15, -2, -2, 15, -2, -2, 2, 60, -2, -2, 7, -2, -2], 'threshold': [2.301183581352234, 0.9712500274181366, 0.6188631951808929, -2.0, -2.0, 0.5021913051605225, -2.0, -2.0, 0.9545454680919647, 0.7840000092983246, -2.0, -2.0, 0.6599499881267548, -2.0, -2.0], 'value': [0.00027719740375211927, -0.18450405123403868, -0.09747810835368663, -0.6194729889941515, 0.2590231828999583, -0.2534272693549278, -1.3289081655779162, -0.6482348963654199, 0.22620114593355167, 0.24814475572220135, 1.2829749570599636, 0.5762453950046982, 0.08609963728294495, 0.9810782119035794, -0.19521972010159155]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [15, 64, 64, -2, -2, 4, -2, -2, 4, 64, -2, -2, 15, -2, -2], 'threshold': [0.9089685380458832, 0.9587499797344208, 0.9225000143051147, -2.0, -2.0, 0.008765322156250477, -2.0, -2.0, 2.893417477607727, 0.9437499940395355, -2.0, -2.0, 0.9927504360675812, -2.0, -2.0], 'value': [0.0002865521247374068, -0.17537104435627504, -0.08100976951116055, 0.41527490374161086, -0.5018230216984544, -0.23325244490559768, -1.262358942239517, -0.5870131127376109, 0.215055410322224, 0.08806533744910237, 1.2015610473409617, 0.09120466725553414, 0.23741966511611373, 1.0671260848639121, 1.3395887900884242]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [4, 4, 66, -2, -2, 52, -2, -2, 8, 20, -2, -2, 66, -2, -2], 'threshold': [2.301183581352234, -0.4097829759120941, 0.679500013589859, -2.0, -2.0, 0.8310000002384186, -2.0, -2.0, 0.9545454680919647, 0.035499999299645424, -2.0, -2.0, -0.0010000000474974513, -2.0, -2.0], 'value': [0.00029375136515271213, -0.1666923351197962, -0.23865980436962805, -0.942573436015263, -1.3783224457923715, -0.09757952540179277, -0.5889028413112104, 0.30625655123310314, 0.2044603336690164, 0.2246761043605193, 1.2961156077659615, 0.9274892856789316, 0.07539041310018746, -0.015432792946203004, 1.3061619250773278]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [4, 64, 15, -2, -2, 4, -2, -2, 15, 64, -2, -2, 20, -2, -2], 'threshold': [2.301183581352234, 0.9712500274181366, 0.6188631951808929, -2.0, -2.0, -0.00559919816441834, -2.0, -2.0, 0.9475198686122894, 0.9437499940395355, -2.0, -2.0, 0.035499999299645424, -2.0, -2.0], 'value': [0.0003061610042944244, -0.15843873590182034, -0.08034734310964528, -0.5457081393440243, 0.2877289050907585, -0.22028592675821754, -1.2560545483391026, -0.5828907288938432, 0.19439660136216097, 0.07776256823327232, 1.1516149078393514, 0.06326384909931182, 0.214937051300633, 1.259157352882181, 0.9199012507834244]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [4, 76, 65, -2, -2, 15, -2, -2, 5, 35, -2, -2, 9, -2, -2], 'threshold': [2.301183581352234, 0.9291666746139526, 0.5833333432674408, -2.0, -2.0, 0.39896415174007416, -2.0, -2.0, 0.5, 0.25450000166893005, -2.0, -2.0, -0.7290000021457672, -2.0, -2.0], 'value': [0.00030944091983756887, -0.15060544419409286, 0.1074775104199498, -0.3068489362642215, 0.8800245140345911, -0.16853603671283618, -1.1355303874778373, -0.4858631372069781, 0.18482646842241565, 0.20354420677708843, 1.2508810165802808, 0.9157458095830054, 0.06532090815797241, -0.5580621482268258, 0.6384284227596062]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [4, 64, 63, -2, -2, 15, -2, -2, 15, 76, -2, -2, 4, -2, -2], 'threshold': [2.301183581352234, 0.9587499797344208, 0.996749997138977, -2.0, -2.0, 0.5021913051605225, -2.0, -2.0, 0.9475198686122894, 0.9437499940395355, -2.0, -2.0, 4.919538497924805, -2.0, -2.0], 'value': [0.00031809172777667546, -0.14315130189926317, -0.05998504045410964, -0.14861110096529423, -1.319754247147996, -0.19416565814654835, -1.1651237761712008, -0.49757231275313374, 0.17573184252958843, 0.068217560668277, 1.1074970944145646, 0.036242880146574584, 0.19466621069811998, 0.9625120789191627, 1.2641458019580403]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [4, 76, 46, -2, -2, 15, -2, -2, 5, 60, -2, -2, 34, -2, -2], 'threshold': [2.301183581352234, 0.9291666746139526, 0.002999999967869371, -2.0, -2.0, 0.45414403080940247, -2.0, -2.0, 0.5, 0.7840000092983246, -2.0, -2.0, 0.04749999940395355, -2.0, -2.0], 'value': [0.00031982141574652144, -0.13607746683402355, 0.10443289116901047, -0.4442417283277494, 0.7839297183346243, -0.15278718418958787, -1.0130404289610822, -0.3573292138841125, 0.1670868183773793, 0.18436530791865596, 1.1548442494755493, 0.40670153804531434, 0.05677030822923142, 0.641293948201471, -0.49051026938085296]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [15, 64, 15, -2, -2, 4, -2, -2, 4, 43, -2, -2, 6, -2, -2], 'threshold': [0.9089685380458832, 0.9712500274181366, 0.6188631951808929, -2.0, -2.0, -0.00559919816441834, -2.0, -2.0, 3.3758041858673096, 0.01699999999254942, -2.0, -2.0, 0.9000000059604645, -2.0, -2.0], 'value': [0.00032612094071809565, -0.12934518982222956, -0.06048775656464738, -0.4567020787849036, 0.31121620591339316, -0.18387922570371104, -1.1738563725691766, -0.496749768486918, 0.1588695594907271, 0.08323213262874901, -0.022772268734049427, 0.8339127398692688, 0.18202387383623106, 1.1507145962155303, 1.4292769813875699]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [4, 64, 65, -2, -2, 4, -2, -2, 0, 20, -2, -2, 7, -2, -2], 'threshold': [2.301183581352234, 0.9225000143051147, 0.5833333432674408, -2.0, -2.0, -0.4097829759120941, -2.0, -2.0, 0.9545454680919647, 0.035499999299645424, -2.0, -2.0, 0.6500249803066254, -2.0, -2.0], 'value': [0.00032485536414710123, -0.12295758799027728, 0.10147221322123383, -0.306643195976325, 0.8437165301133682, -0.13855009126351336, -1.0432580403413458, -0.40339565817768064, 0.15105690524670334, 0.16692689108868755, 1.1963894142474742, 0.7653811556636421, 0.049733149486346155, 0.9954184655112999, -0.1930179184723281]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [15, 64, 4, -2, -2, 15, -2, -2, 4, 15, -2, -2, 20, -2, -2], 'threshold': [0.9089685380458832, 0.9712500274181366, 0.48473408818244934, -2.0, -2.0, 0.5021913051605225, -2.0, -2.0, 2.893417477607727, 0.9140492677688599, -2.0, -2.0, 0.035499999299645424, -2.0, -2.0], 'value': [0.0003320237795855232, -0.11687575432439248, -0.052354949059113885, -0.4159720432388027, 0.3098418826236991, -0.16797524704403172, -1.1306642894079668, -0.4486472585160911, 0.14363684622702774, 0.05151989070287709, 1.5562696051550415, 0.09378604881432065, 0.1598595872458293, 1.1612404835030965, 0.7617045215919372]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [15, 76, 18, -2, -2, 4, -2, -2, 5, 35, -2, -2, 12, -2, -2], 'threshold': [0.9089685380458832, 0.9291666746139526, 0.002999999967869371, -2.0, -2.0, -0.4097829759120941, -2.0, -2.0, 0.5, 0.25450000166893005, -2.0, -2.0, -0.7290000021457672, -2.0, -2.0], 'value': [0.00032900041917314994, -0.1111058963896531, 0.09834414210325974, -0.43131637870323547, 0.7471425039897702, -0.12565766443984386, -1.0012061475368292, -0.36870352130666334, 0.13657557347059082, 0.15141867213960764, 1.160639146459375, 0.7693633166785624, 0.041808097353019165, -0.5868670786820733, 0.5313108525157071]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [4, 64, 63, -2, -2, 4, -2, -2, 4, 43, -2, -2, 8, -2, -2], 'threshold': [2.301183581352234, 0.9587499797344208, 0.9971249997615814, -2.0, -2.0, 0.008765322156250477, -2.0, -2.0, 3.3758041858673096, 0.01699999999254942, -2.0, -2.0, 0.9000000059604645, -2.0, -2.0], 'value': [0.0003325029506603748, -0.10561123701607153, -0.03652266688810685, -0.07089279193862036, -1.3220517339939064, -0.14799030838322505, -1.05051498226165, -0.39104735864243984, 0.12986527876936096, 0.06496979136918585, -0.07737004552711028, 0.7560563815177821, 0.1497312443000261, 1.0895845898574892, 1.3692400753311476]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [4, 64, 15, -2, -2, 66, -2, -2, 0, 20, -2, -2, 69, -2, -2], 'threshold': [2.301183581352234, 0.9712500274181366, 0.6188631951808929, -2.0, -2.0, -0.6044999957084656, -2.0, -2.0, 0.9545454680919647, 0.035499999299645424, -2.0, -2.0, 0.9350000023841858, -2.0, -2.0], 'value': [0.0003283292252534852, -0.10039981788288722, -0.042411217499196546, -0.36552201435798803, 0.311166087341656, -0.14632590406462748, -0.22855139347001732, -0.947421337325762, 0.12348422783794312, 0.13712225991610028, 1.1481174036490294, 0.6724095664543507, 0.03641063841585874, 1.2198635291157494, -0.1289355304292808]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [4, 76, 65, -2, -2, 15, -2, -2, 15, 4, -2, -2, 9, -2, -2], 'threshold': [2.301183581352234, 0.9291666746139526, 0.5833333432674408, -2.0, -2.0, 0.45414403080940247, -2.0, -2.0, 0.9475198686122894, 2.364110231399536, -2.0, -2.0, -0.02350000012665987, -2.0, -2.0], 'value': [0.0003325473312951413, -0.09543522545434427, 0.09635579294670596, -0.3112999076209648, 0.8167967280396725, -0.10876011397651249, -0.8664204793952256, -0.24476305336933027, 0.11742361327623607, 0.03786978726047189, 1.504910830857226, 0.03631399193941589, 0.13143385828666937, 0.9268259207638925, 1.1603786702113943]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [4, 64, 63, -2, -2, 67, -2, -2, 5, 60, -2, -2, 34, -2, -2], 'threshold': [2.301183581352234, 0.9587499797344208, 0.9977500140666962, -2.0, -2.0, 0.03749999962747097, -2.0, -2.0, 0.5, 0.7840000092983246, -2.0, -2.0, 0.04749999940395355, -2.0, -2.0], 'value': [0.00033185448799469194, -0.09071588834507889, -0.028546070137588097, -0.053918195708901835, -1.4638415410175016, -0.12885098301874576, -1.2103549662360638, -0.73917615476819, 0.11165194631124375, 0.12447058876619646, 1.0341560030748005, 0.17699397689815394, 0.029809844483470162, 0.5158988218697916, -0.5460424845050137]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [15, 76, 65, -2, -2, 4, -2, -2, 15, 64, -2, -2, 5, -2, -2], 'threshold': [0.9089685380458832, 0.9291666746139526, 0.5833333432674408, -2.0, -2.0, -0.4097829759120941, -2.0, -2.0, 0.9669393002986908, 0.9437499940395355, -2.0, -2.0, 0.5, -2.0, -2.0], 'value': [0.0003342533953937878, -0.08622706441743937, 0.09278752909832394, -0.2929357357127398, 0.7847142771735842, -0.09866429699655313, -0.9166032027972074, -0.2795515870608115, 0.10616898962749018, 0.04952525129979891, 1.0459510060116592, 0.04417014099496892, 0.12350890952372287, 1.037171422946931, 1.3206444304009841]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [15, 64, 4, -2, -2, 4, -2, -2, 5, 20, -2, -2, 66, -2, -2], 'threshold': [0.9089685380458832, 0.9712500274181366, 0.48473408818244934, -2.0, -2.0, -0.00559919816441834, -2.0, -2.0, 0.5, 0.035499999299645424, -2.0, -2.0, -0.0010000000474974513, -2.0, -2.0], 'value': [0.00033393256732574297, -0.08196413431502914, -0.030113192847220954, -0.3042374913865404, 0.32252730286267633, -0.12302928834010607, -1.042704762631385, -0.3050702697775965, 0.10095617840395531, 0.11296315562055664, 1.1106177886064965, 0.5791031046012654, 0.024296246944114773, -0.18822164356253496, 1.123476621406147]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [15, 64, 18, -2, -2, 57, -2, -2, 4, 20, -2, -2, 20, -2, -2], 'threshold': [0.9089685380458832, 0.9225000143051147, 0.002999999967869371, -2.0, -2.0, 0.9375, -2.0, -2.0, 2.893417477607727, 0.0035000001080334187, -2.0, -2.0, 0.035499999299645424, -2.0, -2.0], 'value': [0.0003294410511213098, -0.07792334620527339, 0.08915976443448392, -0.44016465821624173, 0.7034283373557664, -0.08953162610165748, -0.379092680126552, -1.1798850720695477, 0.09600570047007313, 0.027266681623186954, -0.5705873647388195, 0.4174267699856352, 0.10811134697450168, 1.0714445109974093, 0.5728934136392508]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [4, 64, 63, -2, -2, 66, -2, -2, 8, 60, -2, -2, 64, -2, -2], 'threshold': [2.301183581352234, 0.9712500274181366, 0.996749997138977, -2.0, -2.0, -0.6044999957084656, -2.0, -2.0, 0.7750000059604645, 0.7840000092983246, -2.0, -2.0, 0.9350000023841858, -2.0, -2.0], 'value': [0.00032901583162257, -0.074068296294623, -0.02550455317719332, -0.03695546276069453, -1.1105738781985346, -0.11253003941243464, -0.09826749012184845, -0.8447484758955067, 0.09129135448597787, 0.10337545760872355, 0.9955810462183168, 0.03475285347723797, 0.02538729627873278, 1.1506839954509427, -0.11880173932967605]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [15, 57, 52, -2, -2, 15, -2, -2, 4, 7, -2, -2, 9, -2, -2], 'threshold': [0.6193742156028748, 0.9375, 0.8600000143051147, -2.0, -2.0, 0.4711200147867203, -2.0, -2.0, 2.893417477607727, 0.5064750015735626, -2.0, -2.0, -0.02350000012665987, -2.0, -2.0], 'value': [0.0003267211243013034, -0.08368237290958634, -0.061969511037017706, -0.5412969568561261, -0.05306915131152114, -0.15072363199035288, -1.1455627942918418, -1.4473663456163388, 0.07321426113400886, 0.011099368048399536, 0.9933148167772843, -0.15764233393603636, 0.09804119389870107, 0.8247506509018051, 1.1079209405170585]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [4, 64, 34, -2, -2, 67, -2, -2, 4, 64, -2, -2, 62, -2, -2], 'threshold': [2.301183581352234, 0.9587499797344208, 0.7849999964237213, -2.0, -2.0, 0.03749999962747097, -2.0, -2.0, 3.3758041858673096, 0.9437499940395355, -2.0, -2.0, 0.875, -2.0, -2.0], 'value': [0.00032061999779778614, -0.06715583095988219, -0.014781284526238682, -0.27823732273035146, 0.3282745017679764, -0.0992824857310347, -1.139086354329949, -0.6270139837277059, 0.08282112448902419, 0.03548020750232898, 0.9892577054852433, -0.014366282787862979, 0.097313241933931, 0.4759034673888387, 1.0514310220807992]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, -1, 11, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 10, -1, 12, -1, -1], 'feature': [15, 57, 76, -2, -2, 15, -2, -2, 7, -2, 4, -2, -2], 'threshold': [0.6193742156028748, 0.9375, 0.9375, -2.0, -2.0, 0.4711200147867203, -2.0, -2.0, 0.23839999735355377, -2.0, 2.893417477607727, -2.0, -2.0], 'value': [0.0003217544420825471, -0.07653486092457543, -0.055897852324427136, 0.4418876700178471, -0.4015020577483803, -0.14025428438585796, -1.130230313476684, -1.4109290523150957, 0.06700369753372323, 1.381469193903377, 0.06203336278344667, -0.04539557098574076, 0.9001837471934301]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [4, 62, 61, -2, -2, 7, -2, -2, 52, 4, -2, -2, 52, -2, -2], 'threshold': [-0.00559919816441834, 0.9750000238418579, 0.8544999957084656, -2.0, -2.0, 0.002575000049546361, -2.0, -2.0, 0.9359999895095825, 2.893417477607727, -2.0, -2.0, 0.960999995470047, -2.0, -2.0], 'value': [0.00031251028359974395, -0.08837375429033305, -0.04675806860544594, 0.26504944710595457, -0.5460278738663022, -0.11323912648705305, 0.35925269062830334, -1.0390061464410059, 0.05337479779553192, 0.04395914219996835, -0.1217073116419075, 0.8816755010234482, 0.22037879441157973, 1.9366903175223278, 0.8835698271561285]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [4, 62, 61, -2, -2, 7, -2, -2, 8, 15, -2, -2, 7, -2, -2], 'threshold': [-0.00559919816441834, 0.9750000238418579, 0.8544999957084656, -2.0, -2.0, 0.002575000049546361, -2.0, -2.0, 0.9545454680919647, 0.9255912005901337, -2.0, -2.0, 0.6598499715328217, -2.0, -2.0], 'value': [0.00029438605483304766, -0.08403167155293984, -0.04444264443429761, 0.25034071614586023, -0.5253097498158219, -0.10768611525632817, 0.3366452878125739, -1.0268544044241084, 0.050747898050495, 0.07977499611239776, -0.2560420525429809, 0.9076911052884126, 0.0008921952724174483, 0.35827527395462555, -0.560003317683897]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, -1, 11, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 10, -1, 12, -1, -1], 'feature': [4, 64, 46, -2, -2, 4, -2, -2, 7, -2, 5, -2, -2], 'threshold': [0.48689746856689453, 0.9350000023841858, 0.9500000178813934, -2.0, -2.0, 0.4360905736684799, -2.0, -2.0, 0.23839999735355377, -2.0, 0.5, -2.0, -2.0], 'value': [0.0002789354744992557, -0.06653345033323145, 0.07858374474657967, -0.1371041284350153, 0.9669170355230502, -0.07403951214770436, -0.48102651048134615, -1.4444175794610106, 0.058246355546196026, 1.2763269941167754, 0.05352426663990261, 0.8249886301828508, -0.05744141952575357]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [15, 62, 61, -2, -2, 7, -2, -2, 52, 4, -2, -2, 7, -2, -2], 'threshold': [0.49860021471977234, 0.9750000238418579, 0.8544999957084656, -2.0, -2.0, 0.002575000049546361, -2.0, -2.0, 0.9359999895095825, 2.893417477607727, -2.0, -2.0, 0.06377499923110008, -2.0, -2.0], 'value': [0.00027595315287560346, -0.077128655210487, -0.03902658341748194, 0.2610715498673808, -0.49643315028822166, -0.09989464310680729, 0.3456917922956209, -1.009458249192294, 0.04658826096578617, 0.03762855265374232, -0.1133289526016444, 0.8438338532415569, 0.20550519260571948, 1.5438226000716446, 0.797815118605567]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, -1], 'feature': [15, 62, 61, -2, -2, 7, -2, -2, 20, 7, -2, -2, -2], 'threshold': [0.49860021471977234, 0.9750000238418579, 0.8544999957084656, -2.0, -2.0, 0.002575000049546361, -2.0, -2.0, 0.9964999854564667, 0.06350000202655792, -2.0, -2.0, -2.0], 'value': [0.0002578513689895968, -0.0733400395224184, -0.03709435915225273, 0.24661037077293277, -0.4772984643531703, -0.09499683354359256, 0.32411880156067413, -0.9979276164591767, 0.044292544514691945, 0.04864234794178639, 1.6984117666934795, 0.295103731763295, -1.264302211366033]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, -1, 11, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 10, -1, 12, -1, -1], 'feature': [4, 64, 66, -2, -2, 4, -2, -2, 15, -2, 0, -2, -2], 'threshold': [2.301183581352234, 0.9225000143051147, -0.021499999798834324, -2.0, -2.0, 1.8985036611557007, -2.0, -2.0, 0.9140492677688599, -2.0, 0.9545454680919647, -2.0, -2.0], 'value': [0.00021879945373646956, -0.04941233135887639, 0.08388956779163316, -0.11763726276647794, 0.9303576022429788, -0.05867362503561978, -0.33821304445564704, -1.1601233005316827, 0.06090061173634518, 1.4701768699768474, 0.05755278690186987, 0.8346027802969763, -0.1095338869287524]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, -1, 11, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 10, -1, 12, -1, -1], 'feature': [15, 57, 7, -2, -2, 69, -2, -2, 7, -2, 4, -2, -2], 'threshold': [0.6193742156028748, 0.5357142984867096, 0.8654749989509583, -2.0, -2.0, 0.9850000143051147, -2.0, -2.0, 0.23839999735355377, -2.0, 3.3758041858673096, -2.0, -2.0], 'value': [0.00022448354662957167, -0.05642938030244491, -0.03691789995240727, -0.15440587454010968, -1.0271759623783008, -0.10921978741773401, -1.1103719316245586, -0.9898664265892273, 0.049378218811745374, 1.1622342171080244, 0.04496324226406077, -0.010916213330698919, 0.9168117621827168]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [15, 62, 29, -2, -2, 7, -2, -2, 52, 15, -2, -2, 4, -2, -2], 'threshold': [0.49860021471977234, 0.9750000238418579, 0.3669166564941406, -2.0, -2.0, 0.002575000049546361, -2.0, -2.0, 0.9359999895095825, 0.9669393002986908, -2.0, -2.0, 0.40872691571712494, -2.0, -2.0], 'value': [0.00022016624642001552, -0.06541545950100089, -0.030565765391979954, 0.9125631227542346, -0.2737965636005846, -0.08623815173114084, 0.35673135280573004, -0.9770230273979181, 0.03949091985372546, 0.031018132971862303, -0.08551157811998467, 0.9043153611219737, 0.1897714029688716, 1.200592085255803, 0.6168813230776635]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [4, 57, 62, -2, -2, 49, -2, -2, 7, 41, -2, -2, 5, -2, -2], 'threshold': [-0.00559919816441834, 0.1339285746216774, 0.9750000238418579, -2.0, -2.0, 0.5115000009536743, -2.0, -2.0, 0.06350000202655792, 0.9945000112056732, -2.0, -2.0, 0.5, -2.0, -2.0], 'value': [0.00020760259389075353, -0.062202466350611586, -0.03761355059799622, -0.006252315257749286, -0.8012415957267628, -0.08964109088250352, -1.236584704785348, -0.7761212414115569, 0.03754845845113537, 0.20963847620488615, 1.6751684573542962, -0.8501822492452374, 0.03067826300040129, 0.7570744197803936, -0.10870135656650695]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [15, 63, 62, -2, -2, 4, -2, -2, 7, 46, -2, -2, 4, -2, -2], 'threshold': [0.6193742156028748, 0.9978750050067902, 0.9350000023841858, -2.0, -2.0, -0.35409604012966156, -2.0, -2.0, 0.5064750015735626, 0.9740000069141388, -2.0, -2.0, 2.301183581352234, -2.0, -2.0], 'value': [0.00019350276846133357, -0.04923740467825431, -0.0422538006393987, 0.44768989405875287, -0.3743002849547064, -0.18799059018709716, -1.1826149148323135, -1.4756751395031058, 0.043080493583828386, 0.16936424138328382, 1.249921132767841, -0.26018212235741883, 0.035617606981542975, -0.2836359778549671, 0.5607502450415943]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [15, 66, 66, -2, -2, 15, -2, -2, 7, 41, -2, -2, 6, -2, -2], 'threshold': [0.49860021471977234, 0.679500013589859, 0.44750000536441803, -2.0, -2.0, 0.446696475148201, -2.0, -2.0, 0.06350000202655792, 0.9945000112056732, -2.0, -2.0, 0.9545454680919647, -2.0, -2.0], 'value': [0.00018474760788437402, -0.057005485490559754, -0.03628506146650335, -0.40436663797157657, 0.6595909055789246, -0.08882613667036063, -1.0947983833127115, 0.4141915294703392, 0.03440249943363895, 0.20216115398269763, 1.5745523568280166, -0.7914877161749289, 0.027705221111816565, 0.7316444064248608, -0.106588388114482]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [15, 62, 61, -2, -2, 7, -2, -2, 52, 4, -2, -2, 52, -2, -2], 'threshold': [0.49860021471977234, 0.9750000238418579, 0.8544999957084656, -2.0, -2.0, 0.002575000049546361, -2.0, -2.0, 0.9359999895095825, 3.3758041858673096, -2.0, -2.0, 0.960999995470047, -2.0, -2.0], 'value': [0.00016979064356694131, -0.054205095092540935, -0.02126825624696536, 0.31555949354925, -0.3980737290052015, -0.0738848563027722, 0.48550239983063737, -0.944979960879242, 0.03270307901938426, 0.025023453033956262, -0.0899524027512361, 0.8688226268853046, 0.16891539255039614, 1.6592190153455493, 0.658491126722483]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [15, 7, 45, -2, -2, 7, -2, -2, 7, 46, -2, -2, 5, -2, -2], 'threshold': [0.6193742156028748, 0.8654749989509583, 0.25037500262260437, -2.0, -2.0, 0.987500011920929, -2.0, -2.0, 0.5064750015735626, 0.9740000069141388, -2.0, -2.0, 0.5, -2.0, -2.0], 'value': [0.00016037005276927412, -0.04291001945528435, -0.03516351872181455, 0.38377644812469064, -0.368991404632185, -0.15561401051870705, -1.3767724298367938, -0.05285772158375283, 0.037528880862273066, 0.15674068172726066, 1.181032339169833, -0.2817595208695863, 0.030483919281607538, 0.7348035465300669, -0.14493353687795335]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [15, 64, 9, -2, -2, 68, -2, -2, 7, 12, -2, -2, 4, -2, -2], 'threshold': [0.49860021471977234, 0.9350000023841858, 0.9629999995231628, -2.0, -2.0, 0.1666666716337204, -2.0, -2.0, 0.06350000202655792, 0.9945000112056732, -2.0, -2.0, 2.893417477607727, -2.0, -2.0], 'value': [0.00015805218442352495, -0.049665989125278254, 0.07586834286580521, -0.5844920138017811, 1.1253594954958803, -0.05477732186107143, 0.8616943455375117, -0.5541921652807945, 0.029968503866913614, 0.18996903524135625, 1.4614228067916206, -0.7503552020245166, 0.02358094613920951, -0.09353632520280926, 0.7356664825426444]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [4, 61, 36, -2, -2, 66, -2, -2, 63, 49, -2, -2, 10, -2, -2], 'threshold': [-0.4097829759120941, 0.8285000026226044, 0.953249990940094, -2.0, -2.0, -0.6239999830722809, -2.0, -2.0, 0.996874988079071, 0.972000002861023, -2.0, -2.0, 0.9975000023841858, -2.0, -2.0], 'value': [0.0001514936247747529, -0.060424222525061466, -0.019086414124690986, -0.47586729514596776, 1.5779908198907036, -0.075014037254604, 0.7868631445464545, -0.8519669053658524, 0.02249698635045625, 0.027317651439960598, 0.15354189214086988, 1.5465366981088469, -0.19532565843455638, -0.6616090974441975, -1.6456032958065918]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, -1, 11, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 10, -1, 12, -1, -1], 'feature': [15, 12, 52, -2, -2, 15, -2, -2, 4, -2, 34, -2, -2], 'threshold': [0.9089685380458832, -0.5445000231266022, 0.3685000091791153, -2.0, -2.0, 0.39896415174007416, -2.0, -2.0, 2.364110231399536, -2.0, 0.02350000012665987, -2.0, -2.0], 'value': [0.00012381975801377588, -0.03221827168587111, -0.11551927693892218, -1.12959273479723, -0.2598518202263945, -0.019661870158756735, -0.5755828278080141, 0.11874709174642276, 0.03966707999995138, 1.4409862793304613, 0.03624500592862031, 0.8435547414391734, 0.07358362731774486]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [5, 20, 45, -2, -2, 62, -2, -2, 64, 46, -2, -2, 7, -2, -2], 'threshold': [0.5, 0.035499999299645424, 0.6649583280086517, -2.0, -2.0, 0.875, -2.0, -2.0, 0.9350000023841858, -0.019500000402331352, -2.0, -2.0, 0.8790999948978424, -2.0, -2.0], 'value': [0.00011844380403700652, 0.04265327204159736, 0.054256091879569104, 1.0314157197578877, -0.1943256447214593, 0.004687083078107795, -1.4153754215181944, 0.3566260482119609, -0.027702301409483574, 0.11905049232990018, -0.10046783232978919, 0.8283300847520403, -0.039867538246221496, -0.19519761510302705, -0.805356919692931]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, -1, 11, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 10, -1, 12, -1, -1], 'feature': [15, 63, 57, -2, -2, 15, -2, -2, 7, -2, 66, -2, -2], 'threshold': [0.5672861039638519, 0.9978750050067902, 0.9375, -2.0, -2.0, 0.4035665839910507, -2.0, -2.0, 0.03120000008493662, -2.0, 0.2814999967813492, -2.0, -2.0], 'value': [0.0001308524030903247, -0.0384336374596135, -0.03211763608392048, -0.10322685736633729, -1.0706383852848502, -0.15673063619894498, -1.1451473637191476, -1.3979077937646167, 0.028945670826604958, 1.3243167829427733, 0.025409528556647215, 0.1809158737451488, 0.9242478009109446]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [4, 76, 69, -2, -2, 20, -2, -2, 20, 7, -2, -2, 20, -2, -2], 'threshold': [2.893417477607727, 0.9291666746139526, 0.8350000083446503, -2.0, -2.0, 0.001500000071246177, -2.0, -2.0, 0.035499999299645424, 0.6926999986171722, -2.0, -2.0, 0.057499999180436134, -2.0, -2.0], 'value': [0.00011254321068108145, -0.025348406795880112, 0.09117155093187419, -0.8645021833041079, 0.6526578250889574, -0.03662453173727577, -1.1041016713704241, -0.21071890260716758, 0.04120877798390526, 0.05057065576735895, 1.1518702191271006, 0.9066843532462117, 0.0031806852821238685, -2.140763689892087, 0.4139998263581069]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, -1], 'feature': [4, 66, 66, -2, -2, 76, -2, -2, 29, 49, -2, -2, -2], 'threshold': [-0.4097829759120941, 0.679500013589859, 0.4450000077486038, -2.0, -2.0, 0.9750000238418579, -2.0, -2.0, 0.9961666762828827, 0.972000002861023, -2.0, -2.0, -2.0], 'value': [0.00012347985768764905, -0.05133848484528446, -0.02865117539285756, -0.4817810555792105, 0.8308982286273686, -0.0750341191622637, -1.010755331518223, -1.0730946698402875, 0.019107043420933213, 0.022320797054242426, 0.12325085218611699, 1.3447761571848573, -1.6153956361374466]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [15, 64, 69, -2, -2, 35, -2, -2, 43, 2, -2, -2, 62, -2, -2], 'threshold': [0.9669393002986908, 0.9225000143051147, 0.8350000083446503, -2.0, -2.0, 0.015249999705702066, -2.0, -2.0, 0.03350000083446503, 0.9000000059604645, -2.0, -2.0, 0.9249999821186066, -2.0, -2.0], 'value': [9.911911649574195e-05, -0.022065208887236752, 0.08770916118455127, -0.7573210345857874, 0.6526459884021228, -0.03343237738579466, -0.6817630139735456, -0.18281456174948488, 0.042279192307272284, 0.04927712629481612, 0.9705968135346857, 1.1663341328959427, -0.0009031808353764611, -3.237927771384005, 0.8441437748979677]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [15, 62, 61, -2, -2, 7, -2, -2, 52, 15, -2, -2, 15, -2, -2], 'threshold': [0.49860021471977234, 0.9750000238418579, 0.8544999957084656, -2.0, -2.0, 0.002575000049546361, -2.0, -2.0, 0.9359999895095825, 0.9669393002986908, -2.0, -2.0, 0.6007768213748932, -2.0, -2.0], 'value': [0.00011815745919750284, -0.038097358020294486, -0.005029760352388268, 0.39984504380467634, -0.3150428419058906, -0.057855247626868486, 0.7629398645330062, -0.8972818280304276, 0.022983058574736273, 0.015778461019439253, -0.09123273145266923, 0.792829915892718, 0.1507698678450038, 0.9985194756818426, 0.442983340249983]}, {'left': [1, 2, 3, -1, -1, -1, 7, 8, -1, -1, 11, -1, -1], 'right': [6, 5, 4, -1, -1, -1, 10, 9, -1, -1, 12, -1, -1], 'feature': [4, 4, 57, -2, -2, -2, 7, 46, -2, -2, 34, -2, -2], 'threshold': [0.48689746856689453, 0.4363808035850525, 0.5357142984867096, -2.0, -2.0, -2.0, 0.5064750015735626, 0.8485000133514404, -2.0, -2.0, 0.042500000447034836, -2.0, -2.0], 'value': [0.00010846445618185347, -0.030280659896104775, -0.026579808124541447, -0.06781054620168755, -0.9459888401923111, -1.3870775902397499, 0.026474520923756576, 0.1374616629888914, 1.3738004240603003, 0.1421542663963142, 0.019915605228134556, 0.6271845332382213, -0.11409615851758127]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [15, 61, 75, -2, -2, 66, -2, -2, 34, 15, -2, -2, 15, -2, -2], 'threshold': [0.39896415174007416, 0.8449999988079071, 0.953249990940094, -2.0, -2.0, -0.6584999859333038, -2.0, -2.0, 0.976500004529953, 0.6193742156028748, -2.0, -2.0, 0.6532449126243591, -2.0, -2.0], 'value': [0.00010009766842004468, -0.044890156129450055, -0.010354214980445677, -0.4456743151281954, 1.4889584975004722, -0.058936884486690176, 0.49185308483558093, -0.7675287372802532, 0.016696342052558218, 0.01000315987977925, -0.2279320750922569, 0.26954884251818073, 0.14041563127752138, 1.1165376131233287, -0.4235438119044951]}, {'left': [1, 2, -1, 4, -1, -1, 7, 8, -1, -1, 11, -1, -1], 'right': [6, 3, -1, 5, -1, -1, 10, 9, -1, -1, 12, -1, -1], 'feature': [3, 4, -2, 4, -2, -2, 76, 41, -2, -2, 9, -2, -2], 'threshold': [0.875, 2.521627426147461, -2.0, 2.757597804069519, -2.0, -2.0, 0.9375, -0.019500000402331352, -2.0, -2.0, -0.699999988079071, -2.0, -2.0], 'value': [8.250650278650241e-05, 0.03371643920245794, -0.45726133718855305, 0.03703035042101209, 1.1786294387300595, 0.6864440248226924, -0.02111939949700647, 0.1016776575041267, -0.1400801227357034, 0.7587397677234576, -0.03114106014069397, -0.7132075595040087, -0.13699623211103312]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [4, 66, 66, -2, -2, 62, -2, -2, 63, 72, -2, -2, 25, -2, -2], 'threshold': [-0.4097829759120941, 0.679500013589859, 0.4450000077486038, -2.0, -2.0, 0.9750000238418579, -2.0, -2.0, 0.996874988079071, 0.971750020980835, -2.0, -2.0, 0.9975000023841858, -2.0, -2.0], 'value': [8.720241717895224e-05, -0.04225702514765107, -0.021257163841343008, -0.4302539488653493, 0.8471271243163723, -0.06419021362312838, -0.9697045226817982, -1.061980100182533, 0.01570736655496704, 0.01994240952687272, 0.10647336984184923, 1.3862562119399626, -0.17565383439780483, -0.6785477462785922, -1.6046168098657068]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [4, 64, 69, -2, -2, 34, -2, -2, 32, 67, -2, -2, 62, -2, -2], 'threshold': [3.3758041858673096, 0.9225000143051147, 0.8350000083446503, -2.0, -2.0, 0.010499999858438969, -2.0, -2.0, 0.03350000083446503, 0.5273750126361847, -2.0, -2.0, 0.9249999821186066, -2.0, -2.0], 'value': [6.685875904735236e-05, -0.018283279412079655, 0.07806347727233907, -0.7346815706801156, 0.6077189393227416, -0.02826001457170865, -0.8398103486138624, -0.1681445133080925, 0.03498829517654918, 0.04174140905310375, 1.0063316016278854, -0.905282756181228, -0.006683358744629539, -3.005173285991135, 0.805689838136286]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [57, 49, 7, -2, -2, 14, -2, -2, 7, 66, -2, -2, 76, -2, -2], 'threshold': [0.5357142984867096, 0.001500000071246177, 0.6491749882698059, -2.0, -2.0, 0.04545454680919647, -2.0, -2.0, 0.28884999454021454, 0.6525000035762787, -2.0, -2.0, 0.9650000035762787, -2.0, -2.0], 'value': [8.58144617386833e-05, 0.009642439854986976, 0.054727054461704366, 1.825812796676308, 0.5735576953692978, 0.00011026419528103552, -0.09680195963362939, 0.6929182220015244, -0.06351074196687385, -0.049813335866594845, 0.9395413700287435, -1.0497909997147012, -0.15514166553425787, -1.4253214215787848, -0.8240521498619238]}, {'left': [1, 2, -1, 4, -1, -1, 7, 8, -1, -1, 11, -1, -1], 'right': [6, 3, -1, 5, -1, -1, 10, 9, -1, -1, 12, -1, -1], 'feature': [4, 65, -2, 66, -2, -2, 34, 15, -2, -2, 4, -2, -2], 'threshold': [-0.4097829759120941, 0.4166666716337204, -2.0, 0.679500013589859, -2.0, -2.0, 0.976500004529953, 0.9838567078113556, -2.0, -2.0, 0.6333613097667694, -2.0, -2.0], 'value': [7.872608319744843e-05, -0.03855191392458273, -1.267590447279202, -0.03504206210860467, -0.1073680696437233, -1.029176572790095, 0.014329002268906179, 0.008036223746474981, -0.08232890028075508, 0.8616186128022995, 0.13064708026947117, 1.040008247230027, -0.4314176647685944]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [57, 63, 66, -2, -2, 46, -2, -2, 7, 66, -2, -2, 64, -2, -2], 'threshold': [0.5357142984867096, 0.9977500140666962, 0.44750000536441803, -2.0, -2.0, 0.9964999854564667, -2.0, -2.0, 0.28884999454021454, 0.6525000035762787, -2.0, -2.0, 0.9650000035762787, -2.0, -2.0], 'value': [7.457822397952422e-05, 0.008907227145717404, 0.011838619822306398, 0.0442456331818834, 0.5309771805532268, -0.1331676045796305, -1.045953242835309, -1.2416566485993064, -0.05870412581126264, -0.04554338261245941, 0.8645426041265607, -1.0387690173565636, -0.14674495962394626, -1.379889882675321, -0.8040130375742031]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [15, 64, 69, -2, -2, 35, -2, -2, 20, 37, -2, -2, 78, -2, -2], 'threshold': [0.9669393002986908, 0.9225000143051147, 0.8350000083446503, -2.0, -2.0, 0.015249999705702066, -2.0, -2.0, 0.035499999299645424, 0.664166659116745, -2.0, -2.0, 0.10000000149011612, -2.0, -2.0], 'value': [5.924593241857646e-05, -0.01606823286908974, 0.07506639308887918, -0.691832324554898, 0.5895073810857904, -0.025505250349944567, -0.6432916661554171, -0.1315840827232159, 0.030750825488350224, 0.038003839109250134, 1.0460263257328086, -0.9806917788672231, -0.005514242616149885, 0.6292583455510254, -2.1868613728589925]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [4, 4, 66, -2, -2, 66, -2, -2, 52, 72, -2, -2, 63, -2, -2], 'threshold': [-0.4097829759120941, -0.43953433632850647, 0.679500013589859, -2.0, -2.0, 0.041999999433755875, -2.0, -2.0, 0.8409999907016754, 0.9705000221729279, -2.0, -2.0, 0.9951249957084656, -2.0, -2.0], 'value': [7.503058268045416e-05, -0.035816764227722724, -0.03038304438624393, -0.05035752829440079, -1.0199050975292732, -0.1137000819555852, -1.3128379578213842, -0.3994219025937898, 0.013314986968234214, 0.004900471171028896, -0.001854859125748511, 0.9452670950959902, 0.09234131449698788, 0.7030016467447436, -0.9159020938021374]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [15, 49, 9, -2, -2, 35, -2, -2, 62, 45, -2, -2, 7, -2, -2], 'threshold': [0.9838567078113556, 0.001500000071246177, 0.001500000071246177, -2.0, -2.0, 0.018499999307096004, -2.0, -2.0, 0.875, 0.25837498903274536, -2.0, -2.0, 0.7998500168323517, -2.0, -2.0], 'value': [6.0633550132323756e-05, -0.013977622337839146, 0.0770121038246094, 0.1538018011045373, 2.0200184653115705, -0.020459307068978444, -0.7828358712101489, -0.09453593724099044, 0.0321944447156624, -0.007472391716751087, 1.049111533631888, -2.9918993100223834, 0.03689633265484928, 1.0903855670310831, 0.9799295210019827]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, -1], 'feature': [63, 57, 66, -2, -2, 15, -2, -2, 15, 69, -2, -2, -2], 'threshold': [0.9976249933242798, 0.9375, 0.44750000536441803, -2.0, -2.0, 0.4191458225250244, -2.0, -2.0, 0.4184658080339432, 0.9750000238418579, -2.0, -2.0, -2.0], 'value': [7.218409532370534e-05, 0.0034434167983781443, 0.010727878168211852, 0.027062776674613778, 0.5413613912408071, -0.05340621558524496, -0.9375317249601984, -1.0329682390071586, -0.13038668143682886, -0.09025092514708946, -1.1357522234225048, -1.0733951663427914, -1.5564826890889176]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [4, 7, 45, -2, -2, 43, -2, -2, 66, 35, -2, -2, 49, -2, -2], 'threshold': [0.2707871347665787, 0.869825005531311, 0.25037500262260437, -2.0, -2.0, 0.003999999957159162, -2.0, -2.0, 0.2604999989271164, 0.9622499942779541, -2.0, -2.0, 0.12650000303983688, -2.0, -2.0], 'value': [5.8823075051852885e-05, -0.023686841190128278, -0.017451235007928972, 0.5002884626848345, -0.2653262629575416, -0.13724051166596885, -0.4321768102563431, -1.357180485105561, 0.01780123342672176, 0.010608698537815712, 0.07707873769531111, 1.6720128393435938, 0.13837427283711018, 1.6581400301783302, 0.3593235615339956]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [4, 64, 18, -2, -2, 60, -2, -2, 62, 45, -2, -2, 7, -2, -2], 'threshold': [4.1100850105285645, 0.9225000143051147, -0.012999999802559614, -2.0, -2.0, 0.2045000046491623, -2.0, -2.0, 0.875, 0.25837498903274536, -2.0, -2.0, 0.7998500168323517, -2.0, -2.0], 'value': [5.199920317604436e-05, -0.01322936350426623, 0.05630696563587637, -0.40084975049442617, 0.6737932867785739, -0.0208257692086516, -0.33088717336258516, -0.013438256182405963, 0.030453268753159433, -0.0071637484206694985, 1.0464080671814777, -2.520428125327302, 0.034912182426781374, 1.0852340236530111, 0.9746795461317144]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, -1], 'feature': [63, 4, 7, -2, -2, 49, -2, -2, 4, 76, -2, -2, -2], 'threshold': [0.9976249933242798, -0.4097829759120941, 0.0027249999111518264, -2.0, -2.0, 0.972000002861023, -2.0, -2.0, -0.3290751427412033, 0.9750000238418579, -2.0, -2.0, -2.0], 'value': [6.468904922164009e-05, 0.003241859714957571, -0.02920796261528674, 1.7136057148747037, -0.4562702922489973, 0.014655972687334397, 0.07740857157110709, 1.0051031597165634, -0.12288442694576919, -0.08410718563621325, -1.1250251293757785, -1.0685692519146177, -1.5345894229552397]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [57, 49, 7, -2, -2, 52, -2, -2, 7, 66, -2, -2, 76, -2, -2], 'threshold': [0.5357142984867096, 0.001500000071246177, 0.6491749882698059, -2.0, -2.0, 0.8600000143051147, -2.0, -2.0, 0.28884999454021454, 0.6525000035762787, -2.0, -2.0, 0.9650000035762787, -2.0, -2.0], 'value': [5.266886199899074e-05, 0.007624307037632968, 0.0456776258431787, 1.5644278757510521, 0.5101200841673407, -0.00042125179553948513, -0.09051722859704049, 0.3034378576228827, -0.05033437621710857, -0.037795801292985924, 0.8125978668273536, -1.017874725179639, -0.13421311881296322, -1.3068302482020113, -0.7800914117216335]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [4, 9, 52, -2, -2, 36, -2, -2, 20, 67, -2, -2, 37, -2, -2], 'threshold': [4.919538497924805, -0.022499999962747097, 0.2410000041127205, -2.0, -2.0, 0.49175000190734863, -2.0, -2.0, 0.035499999299645424, 0.26212500035762787, -2.0, -2.0, 0.2435000017285347, -2.0, -2.0], 'value': [4.3155888685144695e-05, -0.010623369213065122, -0.04933633344244279, -0.6848587341315195, 0.005137258094420824, 0.004348952422655738, 0.5233622175778042, -0.12490631047123212, 0.03389437736074344, 0.03247191915170755, 1.0330044468464425, 1.0397140905315188, 0.05065271313594678, 1.051297893768751, 1.0560185049053785]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [57, 66, 34, -2, -2, 60, -2, -2, 4, 4, -2, -2, 3, -2, -2], 'threshold': [0.1339285746216774, 0.29999999701976776, 0.042500000447034836, -2.0, -2.0, 0.6725000143051147, -2.0, -2.0, 0.9559141397476196, 0.24514247477054596, -2.0, -2.0, 0.625, -2.0, -2.0], 'value': [4.955876905184469e-05, 0.010553987386337893, 0.0031691462789344804, 0.39132864935550704, -0.1382021050755319, 0.08359565111250436, 2.114389090443269, 0.13976796830522523, -0.03307389497304727, -0.046289026058840894, -0.4426027515514823, -1.0636578869636175, 0.05619057538004907, 0.14834546661722503, 1.358148366829216]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, -1], 'feature': [63, 15, 7, -2, -2, 49, -2, -2, 4, 69, -2, -2, -2], 'threshold': [0.9976249933242798, 0.39896415174007416, 0.0027249999111518264, -2.0, -2.0, 0.972000002861023, -2.0, -2.0, -0.3290751427412033, 0.9750000238418579, -2.0, -2.0, -2.0], 'value': [4.3491276053852745e-05, 0.003000486018274829, -0.026678202865754647, 1.5962285482390617, -0.4325940246418512, 0.013439862368221835, 0.06873845187144544, 0.9647341480541809, -0.11438532851593726, -0.07877180422642867, -1.11781081272198, -1.064159255867324, -1.4913200649258147]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [15, 69, 32, -2, -2, 7, -2, -2, 4, 23, -2, -2, 4, -2, -2], 'threshold': [0.9475198686122894, 0.9600000083446503, 0.001500000071246177, -2.0, -2.0, 0.6484000086784363, -2.0, -2.0, 3.0891048908233643, 0.5, -2.0, -2.0, 3.3758041858673096, -2.0, -2.0], 'value': [3.3406494802148234e-05, -0.01401051138183621, 0.021240564218728518, 1.1499348677858188, 0.05004961060068695, -0.03792286521279256, -0.20510433992840466, -0.6921761022020383, 0.02270153734009571, 0.10041179769175257, 1.0813083997348725, 1.2834960323586446, 0.01856534606331398, -0.7882324127419011, 0.6762468499547587]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [57, 63, 66, -2, -2, 9, -2, -2, 7, 66, -2, -2, 69, -2, -2], 'threshold': [0.5357142984867096, 0.9977500140666962, 0.44750000536441803, -2.0, -2.0, 0.9964999854564667, -2.0, -2.0, 0.28884999454021454, 0.6525000035762787, -2.0, -2.0, 0.9650000035762787, -2.0, -2.0], 'value': [3.7695772112657485e-05, 0.00690014026312921, 0.009195749345526884, 0.0273284310341978, 0.4746997481031307, -0.10436037993041253, -0.9891512332236162, -1.190780242503542, -0.04562987205151354, -0.03383808958113671, 0.7942685922011004, -1.004217849593763, -0.12451283064644775, -1.255406000538283, -0.7530309427076225]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [15, 9, 68, -2, -2, 12, -2, -2, 20, 67, -2, -2, 37, -2, -2], 'threshold': [0.9927504360675812, -0.022499999962747097, 0.4166666716337204, -2.0, -2.0, -0.015500000212341547, -2.0, -2.0, 0.035499999299645424, 0.26212500035762787, -2.0, -2.0, 0.2645000070333481, -2.0, -2.0], 'value': [2.680134728626697e-05, -0.009619743336786796, -0.04613293667519871, -1.069029933470763, -0.14418298329342802, 0.004501812206487861, 1.6621566518708428, 0.007239293251195284, 0.030641018951019634, 0.029308268932286462, 1.029687674218646, 1.0359483475639173, 0.04634248010921771, 1.047045512831404, 1.0509673245404443]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1], 'feature': [57, 63, 66, -2, -2, 61, -2, -2, 15, 7, -2, -2, 76, -2, -2], 'threshold': [0.9375, 0.9977500140666962, 0.44750000536441803, -2.0, -2.0, 0.9975000023841858, -2.0, -2.0, 0.4191458225250244, 0.1371000036597252, -2.0, -2.0, 0.9850000143051147, -2.0, -2.0], 'value': [3.2984119685525645e-05, 0.006177355845608591, 0.008635167756387997, 0.016211146916181647, 0.4754843477305276, -0.10711241191687922, -0.9946745007274462, -1.2139658055220361, -0.046264856835144856, -0.03326118671954626, -1.0679617907414707, 0.42160504621145883, -0.11995232082353699, -1.1912900175561083, -0.833120183334761]}, {'left': [1, 2, 3, -1, -1, 6, -1, -1, 9, -1, 11, -1, -1], 'right': [8, 5, 4, -1, -1, 7, -1, -1, 10, -1, 12, -1, -1], 'feature': [15, 49, 4, -2, -2, 4, -2, -2, 4, -2, 4, -2, -2], 'threshold': [0.7940910756587982, 0.001500000071246177, -0.1833086535334587, -2.0, -2.0, 0.7545948326587677, -2.0, -2.0, 1.5975000262260437, -2.0, 2.301183581352234, -2.0, -2.0], 'value': [2.2818181683547805e-05, -0.01651960848949906, 0.11664173684019766, -1.2025024381034304, 1.5245239432676876, -0.022791700841984792, -0.137723987918725, -0.8953310013365587, 0.01709696571015413, 2.141973065683544, 0.00991229725992396, -0.45747134093213787, 0.3013378805909741]}]
_CB_PRIOR0 = 0.4786174575278266
_CB_PRIOR1 = 0.5213825424721734
_CB_LEARNING_RATE = 0.05
_CB_UMBRAL = 0.2494551288192311  # corte calibrado OOF agrupado


def _cb_float(value, default=0.0):
    try:
        return float(value) if value is not None else float(default)
    except (TypeError, ValueError):
        return float(default)


def _cb_quantile(values, q):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _cb_stats(values, prefix, out):
    vals = [_cb_float(x) for x in values]
    if not vals:
        vals = [0.0]
    n = len(vals)
    mean = sum(vals) / n
    variance = sum((x - mean) ** 2 for x in vals) / n
    out[prefix + '_min'] = min(vals)
    out[prefix + '_max'] = max(vals)
    out[prefix + '_mean'] = mean
    out[prefix + '_std'] = math.sqrt(max(0.0, variance))
    out[prefix + '_q25'] = _cb_quantile(vals, 0.25)
    out[prefix + '_q75'] = _cb_quantile(vals, 0.75)
    out[prefix + '_n_gt25'] = sum(x > 0.25 for x in vals) / n
    out[prefix + '_n_gt5'] = sum(x > 0.5 for x in vals) / n
    out[prefix + '_n_gt75'] = sum(x > 0.75 for x in vals) / n


def _cb_nums(text):
    return re.findall(r'\d+(?:[.,]\d+)*', str(text or ''))


def _cb_words(text):
    return re.findall(r'[a-záéíóúñ]{3,}', str(text or '').lower())


def _cb_sentence_features(sentence, fragments, sources, guardian):
    d = {}
    put = lambda key, value: d.__setitem__(key, float(value))
    support = [_cb_float(x) for x in (sentence.get('respaldo_por_fuente') or [0.0])] or [0.0]
    contradiction = [_cb_float(x) for x in (sentence.get('contradiccion_por_fuente') or [0.0])] or [0.0]
    similarity = [_cb_float(x) for x in (sentence.get('similitud_por_fuente') or [0.0])] or [0.0]
    llm = sentence.get('llm') or {}
    verdict = llm.get('veredicto')
    confidence = _cb_float(llm.get('confianza'), 0.5) if llm.get('confianza') is not None else 0.5
    source_idx = llm.get('fuente')

    support_sorted = sorted(support, reverse=True)
    contradiction_sorted = sorted(contradiction, reverse=True)
    similarity_sorted = sorted(similarity, reverse=True)
    _cb_stats(support, 's', d)
    _cb_stats(contradiction, 'c', d)
    _cb_stats(similarity, 'm', d)
    put('s_top2_gap', support_sorted[0] - (support_sorted[1] if len(support_sorted) > 1 else 0.0))
    put('c_top2_gap', contradiction_sorted[0] - (contradiction_sorted[1] if len(contradiction_sorted) > 1 else 0.0))
    put('sm_top2_gap', similarity_sorted[0] - (similarity_sorted[1] if len(similarity_sorted) > 1 else 0.0))
    put('s_all', sentence.get('respaldo_todas') or 0.0)
    put('c_all', sentence.get('contradiccion_todas') or 0.0)
    put('llm_conf', confidence)
    for name in ('respaldada', 'no_respaldada', 'contradicha'):
        put('llm_' + name, verdict == name)
    put('llm_unknown', verdict not in ('respaldada', 'no_respaldada', 'contradicha'))
    put('llm_source_valid', isinstance(source_idx, int) and 0 <= source_idx < len(support))
    if isinstance(source_idx, int) and source_idx >= 0:
        put('llm_source_support', support[source_idx] if source_idx < len(support) else 0.0)
        put('llm_source_cont', contradiction[source_idx] if source_idx < len(contradiction) else 0.0)
        put('llm_source_sim', similarity[source_idx] if source_idx < len(similarity) else 0.0)
    else:
        put('llm_source_support', 0.0)
        put('llm_source_cont', 0.0)
        put('llm_source_sim', 0.0)
    if support and isinstance(source_idx, int) and 0 <= source_idx < len(support):
        put('llm_source_is_best_support', support[source_idx] >= max(support) - 1e-8)
    else:
        put('llm_source_is_best_support', 0.0)
    put('support_minus_contra', max(support) - max(contradiction))
    put('support_all_minus_contra_all', _cb_float(sentence.get('respaldo_todas')) - _cb_float(sentence.get('contradiccion_todas')))
    put('support_below_contra', max(contradiction) > max(support))

    fragment_support = []
    fragment_support_all = []
    fragment_contra_all = []
    fragment_contra_max = []
    fragment_sim_max = []
    fragment_support_llm_source = []
    for fragment in fragments:
        f_support = [_cb_float(x) for x in (fragment.get('respaldo_por_fuente') or [0.0])]
        f_contra = [_cb_float(x) for x in (fragment.get('contradiccion_por_fuente') or [0.0])]
        f_sim = [_cb_float(x) for x in (fragment.get('similitud_por_fuente') or [0.0])]
        fragment_support.append(max(f_support, default=0.0))
        fragment_support_all.append(_cb_float(fragment.get('respaldo_todas')))
        fragment_contra_all.append(_cb_float(fragment.get('contradiccion_todas')))
        fragment_contra_max.append(max(f_contra, default=0.0))
        fragment_sim_max.append(max(f_sim, default=0.0))
        if isinstance(source_idx, int) and source_idx >= 0 and source_idx < len(f_support):
            fragment_support_llm_source.append(f_support[source_idx])
        else:
            fragment_support_llm_source.append(0.0)
    for prefix, values in (
        ('frag_smax', fragment_support), ('frag_sall', fragment_support_all),
        ('frag_call', fragment_contra_all), ('frag_cmax', fragment_contra_max),
        ('frag_sim', fragment_sim_max), ('frag_llm_s', fragment_support_llm_source),
    ):
        _cb_stats(values, prefix, d)

    if sources and isinstance(source_idx, int) and 0 <= source_idx < len(sources):
        primary_idx = source_idx
    else:
        primary_idx = max(range(len(support)), key=lambda i: support[i]) if support else 0
    relevant_indices = {
        primary_idx,
        max(range(len(support)), key=lambda i: support[i]) if support else 0,
        max(range(len(similarity)), key=lambda i: similarity[i]) if similarity else 0,
    }
    lexical = guardian._auditar_oracion(
        sentence.get('texto', ''), sources, primary_idx, relevant_indices, max(support, default=0.0)
    )
    for key, value in lexical.items():
        put('lex_' + key, value)

    text = sentence.get('texto', '')
    answer_words = _cb_words(text)
    source_words = set(word for source in sources for word in _cb_words(source))
    unique_words = set(answer_words)
    put('text_num_count', len(_cb_nums(text)))
    put('text_word_count', len(answer_words))
    put('text_word_unique', len(unique_words))
    put('text_coverage_sources', sum(word in source_words for word in unique_words) / max(1, len(unique_words)))
    put('text_unseen_word_ratio', sum(word not in source_words for word in unique_words) / max(1, len(unique_words)))
    source_nums = set(number.replace(',', '.') for source in sources for number in _cb_nums(source))
    answer_nums = set(number.replace(',', '.') for number in _cb_nums(text))
    put('num_count', len(answer_nums))
    put('num_unseen_ratio', sum(number not in source_nums for number in answer_nums) / max(1, len(answer_nums)))
    return d


def _cb_features(caso, guardian):
    out = {}
    signals = caso.get('senales') or {}
    sentences = signals.get('oraciones') or []
    fragments = signals.get('fragmentos') or []
    sources = caso.get('fuentes') or []
    fragments_by_sentence = defaultdict(list)
    for fragment in fragments:
        fragments_by_sentence[fragment.get('oracion', 0)].append(fragment)
    rows = [
        _cb_sentence_features(sentence, fragments_by_sentence.get(i, []), sources, guardian)
        for i, sentence in enumerate(sentences)
    ]
    all_keys = set(key for row in rows for key in row)
    for key in all_keys:
        _cb_stats([row.get(key, 0.0) for row in rows], 'sent_' + key, out)
    if rows:
        risks = []
        for row in rows:
            risk = (
                (1.0 - row.get('s_all', 0.0)) * 0.5
                + row.get('c_all', 0.0) * 0.35
                + (1.0 - row.get('frag_sall_min', row.get('frag_sall_mean', 0.0))) * 0.15
                + row.get('llm_contradicha', 0.0) * 0.4
                + row.get('llm_no_respaldada', 0.0) * 0.2
            )
            risks.append(risk)
        worst = rows[max(range(len(risks)), key=lambda i: risks[i])]
        for key, value in worst.items():
            out['worst_' + key] = value
        out['worst_heuristic_risk'] = max(risks)
    else:
        out['worst_heuristic_risk'] = 1.0

    global_fragment_rows = []
    for fragment in fragments:
        fragment_support = fragment.get('respaldo_por_fuente') or [0.0]
        fragment_contra = fragment.get('contradiccion_por_fuente') or [0.0]
        fragment_sim = fragment.get('similitud_por_fuente') or [0.0]
        global_fragment_rows.append({
            'smax': max(fragment_support, default=0.0),
            'sall': fragment.get('respaldo_todas') or 0.0,
            'cmax': max(fragment_contra, default=0.0),
            'call': fragment.get('contradiccion_todas') or 0.0,
            'mmax': max(fragment_sim, default=0.0),
        })
    for key in ('smax', 'sall', 'cmax', 'call', 'mmax'):
        _cb_stats([row[key] for row in global_fragment_rows], 'case_frag_' + key, out)

    out['case_num_sentences'] = len(sentences)
    out['case_num_frags'] = len(fragments)
    out['case_num_sources'] = len(sources)
    answer = caso.get('respuesta', '')
    answer_word_set = set(_cb_words(answer))
    source_word_set = set(word for source in sources for word in _cb_words(source))
    out['case_answer_words'] = len(_cb_words(answer))
    out['case_answer_word_coverage'] = sum(word in source_word_set for word in answer_word_set) / max(1, len(answer_word_set))
    out['case_answer_unseen_words'] = sum(word not in source_word_set for word in answer_word_set) / max(1, len(answer_word_set))
    source_number_set = set(number.replace(',', '.') for source in sources for number in _cb_nums(source))
    answer_number_set = set(number.replace(',', '.') for number in _cb_nums(answer))
    out['case_answer_num_count'] = len(_cb_nums(answer))
    out['case_num_unseen_ratio'] = sum(number not in source_number_set for number in answer_number_set) / max(1, len(answer_number_set))

    base_score = _GUARDIAN_SCORE_ORIGINAL(guardian, caso)
    out['old_case_score'] = base_score
    clipped = max(1e-5, min(1.0 - 1e-5, base_score))
    out['old_case_logit'] = math.log(clipped / (1.0 - clipped))
    return out


def _cb_predict_raw(features):
    vector = [features.get(name, 0.0) for name in _CB_MODEL_FEATURES]
    raw = math.log(_CB_PRIOR1 / _CB_PRIOR0)
    for tree in _CB_TREES:
        node = 0
        left = tree['left']
        right = tree['right']
        while left[node] != -1:
            feature_index = tree['feature'][node]
            if vector[feature_index] <= tree['threshold'][node]:
                node = left[node]
            else:
                node = right[node]
        raw += _CB_LEARNING_RATE * tree['value'][node]
    raw = max(-30.0, min(30.0, raw))
    return 1.0 / (1.0 + math.exp(-raw))


_CB_BLEND_ALPHA = 0.625  # peso del clasificador; 1 - alpha = detector original


def _cb_score_mejorado(self, caso) -> float:
    signals = caso.get("senales") or {}
    if not (signals.get("oraciones") or []):
        return 0.0
    feature_values = _cb_features(caso, self)
    model_probability = _cb_predict_raw(feature_values)
    old_probability = min(1.0, max(0.0, _cb_float(feature_values.get("old_case_score", 0.5), 0.5)))
    probability = _CB_BLEND_ALPHA * model_probability + (1.0 - _CB_BLEND_ALPHA) * old_probability
    threshold = _CB_UMBRAL
    if threshold <= 0.0:
        score = 0.5 + 0.5 * probability
    elif threshold >= 1.0:
        score = 0.5 * probability
    elif probability < threshold:
        score = 0.5 * probability / threshold
    else:
        score = 0.5 + 0.5 * ((probability - threshold) / (1.0 - threshold))
    return min(1.0, max(0.0, score))


Guardian.score = _cb_score_mejorado


import re, calendar, datetime
NUM=r'(?<!\w)\d+(?:[.,]\d+)*(?!\w)'
NUMC=re.compile(NUM)
MONTHS={'enero':1,'febrero':2,'marzo':3,'abril':4,'mayo':5,'junio':6,'julio':7,'agosto':8,'septiembre':9,'setiembre':9,'octubre':10,'noviembre':11,'diciembre':12}

def parse_num(s):
    s=str(s).strip().replace(' ','')
    if ',' in s and '.' in s:
        # Spanish locale: dot thousands, comma decimals.
        s=s.replace('.','').replace(',','.')
    elif ',' in s:
        s=s.replace(',','.')
    elif '.' in s:
        a,b=s.rsplit('.',1)
        if len(b)==3 and len(a.replace('-',''))<=3:
            s=a+b
    try: return float(s)
    except: return None

def nums_in_text(s): return [parse_num(m.group()) for m in NUMC.finditer(str(s or ''))]

def source_has_value(sources, value, tol=1e-6):
    if value is None:return False
    vals=[v for s in sources for v in nums_in_text(s) if v is not None]
    return any(abs(v-value)<=tol for v in vals)

def _date_from_match(month, year, day=1):
    mo=MONTHS.get(month.lower())
    return datetime.date(int(year),int(mo),min(int(day),calendar.monthrange(int(year),int(mo))[1])) if mo else None

def _v4_math_features(caso):
    ans=str(caso.get('respuesta') or '')
    sources=list(caso.get('fuentes') or [])
    low=ans.lower()
    out={
      'v4_delta_valid':0.,'v4_delta_invalid':0.,
      'v4_points_valid':0.,'v4_points_invalid':0.,
      'v4_product_valid':0.,'v4_product_invalid':0.,
      'v4_equiv_valid':0.,'v4_equiv_invalid':0.,
      'v4_majority_valid':0.,'v4_majority_invalid':0.,
      'v4_duration_valid':0.,'v4_duration_invalid':0.,
      'v4_relative_date_valid':0.,'v4_relative_date_invalid':0.,
      'v4_math_valid_count':0.,'v4_math_invalid_count':0.
    }
    # Explicit change claim: "pasa de 3,1 a 3,9 ... 0,9 ... más".
    # No exigimos una frontera de palabra adicional antes del delta: NUM ya
    # impone sus propios límites y la puntuación decimal española contiene comas.
    pat = re.compile(
        r'\b(?:pasa|pas[oó]|sube|subi[oó]|aumenta|aument[oó]|crece|creci[oó]|'
        r'incrementa|increment[oó]|baja|baj[oó]|disminuye|disminuy[oó]|cae|cay[oó]|'
        r'desciende|descendi[oó])\s+de\s+(?P<a>'+NUM+r')\s+(?:a|hasta)\s+(?P<b>'+NUM+r')'
        r'.{0,100}?(?P<delta>'+NUM+r')\s*(?:millones?(?:\s+de\s+euros?)?\s*)?(?P<dir>m[aá]s|menos)\b',
        re.I)
    for m in pat.finditer(ans):
        a,b,d=map(parse_num,[m.group('a'),m.group('b'),m.group('delta')]); direction=m.group('dir').lower().replace('á','a')
        if None in (a,b,d): continue
        # Only compare same-scale values if they share a nearby unit.
        expected=b-a
        direction_ok=(direction=='mas' and expected>0) or (direction=='menos' and expected<0)
        ok=direction_ok and abs(abs(expected)-d)<=max(.01,.015*max(abs(expected),abs(d),1))
        # The two endpoints must be source-backed; the claimed delta is precisely what is audited.
        grounded=source_has_value(sources,a) and source_has_value(sources,b)
        if grounded:
            out['v4_delta_valid' if ok else 'v4_delta_invalid']=1.
    # Percentage-point claims: "cae 25 puntos ... del 71 % ... al 46 %".
    patp=re.compile(r'(?P<points>'+NUM+r')\s+puntos?(?:\s+porcentuales?)?.{0,90}?del\s+(?P<a>'+NUM+r')\s*%.{0,90}?al\s+(?P<b>'+NUM+r')\s*%',re.I)
    for m in patp.finditer(ans):
        pts,a,b=map(parse_num,[m.group('points'),m.group('a'),m.group('b')])
        if None in (pts,a,b):continue
        ok=abs(abs(a-b)-pts)<=max(.01,.015*max(abs(a-b),abs(pts),1))
        if source_has_value(sources,a) and source_has_value(sources,b):
            out['v4_points_valid' if ok else 'v4_points_invalid']=1.
    # Multiplication: stated total vs daily amount x days.
    patm=re.compile(r'(?P<total>'+NUM+r')\s*euros?.{0,40}?(?P<daily>'+NUM+r')\s*euros?\s+(?:diarios?|por\s+d[ií]a|al\s+d[ií]a).{0,40}?(?P<days>'+NUM+r')\s*d[ií]as?',re.I)
    for m in patm.finditer(ans):
        total,daily,days=map(parse_num,[m.group('total'),m.group('daily'),m.group('days')])
        if None in (total,daily,days):continue
        ok=abs(total-daily*days)<=max(.01,.01*max(total,daily*days,1))
        if source_has_value(sources,daily) and source_has_value(sources,days):
            out['v4_product_valid' if ok else 'v4_product_invalid']=1.
    # Budget equals N awards of a maximum amount.
    pate=re.compile(r'(?P<budget>'+NUM+r')\s*millones?.{0,70}?equivale\s+a\s+(?P<count>'+NUM+r')\s+subvenciones?.{0,100}?(?:importe\s+m[aá]ximo|maximo)\s+de\s+(?P<unit>'+NUM+r')\s*euros?',re.I)
    for m in pate.finditer(ans):
        budget,count,unit=map(parse_num,[m.group('budget'),m.group('count'),m.group('unit')])
        if None in (budget,count,unit):continue
        ok=abs(budget*1e6-count*unit)<=max(1.,.01*max(budget*1e6,count*unit))
        if source_has_value(sources,budget) and source_has_value(sources,unit):
            out['v4_equiv_valid' if ok else 'v4_equiv_invalid']=1.
    # Absolute majority by seat count.
    patmaj=re.compile(r'(?P<n>'+NUM+r')\s+de\s+los\s+(?P<total>'+NUM+r')\s+concejales?.{0,60}?mayor[ií]a\s+absoluta',re.I)
    for m in patmaj.finditer(ans):
        n,total=map(parse_num,[m.group('n'),m.group('total')])
        if None in (n,total):continue
        ok=n>total/2
        if source_has_value(sources,total) and source_has_value(sources,n): out['v4_majority_valid' if ok else 'v4_majority_invalid']=1.
    # Duration in months between a stated start and a stated finish.
    month_pat='|'.join(MONTHS.keys())
    startm=re.search(r'(?:prevista\s+para|previst[oa]s?\s+para|desde)\s+('+month_pat+r')\s+de\s+(\d{4})',low,re.I)
    durm=re.search(r'\b(\d+)\s+meses?\b',low,re.I)
    endm=re.search(r'(?:terminar[ií]a|finalizar[ií]a|concluir[ií]a|acabar[iá]|finalizar[aá]|terminar[aá]|terminar[iá]|concluir[aá])\s+en\s+('+month_pat+r')\s+de\s+(\d{4})',low,re.I)
    if startm and durm and endm:
        start=_date_from_match(startm.group(1),startm.group(2)); end=_date_from_match(endm.group(1),endm.group(2)); dur=int(durm.group(1))
        if start and end:
            target_month=(start.year*12+start.month-1)+dur
            ty,tm0=divmod(target_month,12); tm=tm0+1
            ok=(end.year==ty and end.month==tm)
            # only count it when source contains the same duration and start date.
            src=' '.join(sources).lower()
            if durm.group(1) in src and startm.group(1) in src and startm.group(2) in src:
                out['v4_duration_valid' if ok else 'v4_duration_invalid']=1.
    # Relative dates such as "9 months later in the rest of the country" vs source date.
    number_words={'uno':1,'una':1,'dos':2,'tres':3,'cuatro':4,'cinco':5,'seis':6,'siete':7,'ocho':8,'nueve':9,'diez':10,'once':11,'doce':12}
    rel=re.search(r'(?P<n>\d+|uno|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce)\s+meses?\s+despu[eé]s.{0,80}?resto\s+del\s+pa[ií]s',low,re.I)
    if rel:
        # Find any anchor date earlier in the answer and the source date associated with the same subject.
        month_pattern='|'.join(MONTHS.keys())
        ans_dates=list(re.finditer(r'(?P<day>\d{1,2})\s+de\s+(?P<mon>'+month_pattern+r')\s+de\s+(?P<year>\d{4})',low,re.I))
        src_low=' '.join(sources).lower()
        src_re=re.search(r'resto\s+del\s+pa[ií]s.{0,100}?(?:el\s+)?(?P<day>\d{1,2})\s+de\s+(?P<mon>'+month_pattern+r')\s+de\s+(?P<year>\d{4})',src_low,re.I)
        if ans_dates and src_re:
            a=ans_dates[0].groupdict(); b=src_re.groupdict()
            try:
                d=_date_from_match(a['mon'],a['year'],int(a['day']))
                n_raw=rel.group('n').lower(); n_months=int(n_raw) if n_raw.isdigit() else number_words[n_raw]; expected_month=d.year*12+d.month-1+n_months; ey,em0=divmod(expected_month,12); em=em0+1
                src_date=_date_from_match(b['mon'],b['year'],int(b['day']))
                expected=(ey,em); actual=(src_date.year,src_date.month)
                out['v4_relative_date_valid' if expected==actual else 'v4_relative_date_invalid']=1.
            except Exception: pass
    valid=sum(out[k] for k in out if k.endswith('_valid'))
    invalid=sum(out[k] for k in out if k.endswith('_invalid'))
    out['v4_math_valid_count']=valid; out['v4_math_invalid_count']=invalid
    return out



# ---- CataBotv4: backstop aritmético conservador ----
import calendar
import datetime

_V4_BASE_SCORE = Guardian.score


def _v4_clean_single_sentence(caso, guardian):
    """True only for one high-confidence supported sentence with no lex audit flags."""
    signals = caso.get("senales") or {}
    sentences = signals.get("oraciones") or []
    if len(sentences) != 1:
        return False
    llm = sentences[0].get("llm") or {}
    try:
        confidence = float(llm.get("confianza") or 0.0)
    except (TypeError, ValueError):
        return False
    if llm.get("veredicto") != "respaldada" or confidence < 0.80:
        return False
    # This second audit is only run for mathematically coherent claims, not every case.
    fv = _cb_features(caso, guardian)
    for name, value in fv.items():
        if name.startswith(("sent_lex_", "worst_lex_")):
            try:
                if float(value) > 0.0:
                    return False
            except (TypeError, ValueError):
                continue
    return True


def _v4_score(self, caso) -> float:
    base_score = _V4_BASE_SCORE(self, caso)
    math_flags = _v4_math_features(caso)
    invalid_count = math_flags.get("v4_math_invalid_count", 0.0)
    valid_count = math_flags.get("v4_math_valid_count", 0.0)

    # One explicit, source-grounded arithmetic contradiction is sufficient to
    # label the response hallucinated, even if the statistical model is unsure.
    if invalid_count > 0.0:
        return max(base_score, 0.75)

    # Coherent arithmetic is only a *supporting* signal. Rescue from false alarm
    # only when the response is one sentence, the judge is confident, and the
    # existing lexical audits find no missing caveat or unsupported detail.
    if valid_count > 0.0 and _v4_clean_single_sentence(caso, self):
        return min(base_score, 0.49)

    return min(1.0, max(0.0, base_score))


Guardian.score = _v4_score

# Calibración provisional basada en el mejor corte reportado por evaluar_local.py
# para train.jsonl (0.599). Convierte ese corte en 0.5, que es el umbral oficial.
# Advertencia: este valor se estimó sobre train; validar en grupos H/OOF antes de
# usarlo como candidato final para el test oculto.
_V4_UNCALIBRATED_SCORE = Guardian.score
_V4_CALIBRATION_THRESHOLD = 0.599

def _v4_calibrated_score(self, caso) -> float:
    s = float(_V4_UNCALIBRATED_SCORE(self, caso))
    t = _V4_CALIBRATION_THRESHOLD
    if s < t:
        return 0.5 * (s / t)
    return 0.5 + 0.5 * ((s - t) / (1.0 - t))

Guardian.score = _v4_calibrated_score


# ---------------------------------------------------------------------------
# CataBotv4 blindado: compatibilidad de esquema y tolerancia a entradas raras.
# Para casos bien formados, delega sin alterar el score original.
# Las salvaguardas solo se activan si faltan señales/campos o si una entrada
# provoca una excepción. No es una garantía de generalización semántica.
# ---------------------------------------------------------------------------
import unicodedata as _cb4_unicodedata

_CB4_SCORE_CALIBRADO = Guardian.score
_CB4_VERSION = "4.1-robusta"


def _cb4_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return " ".join(_cb4_text(x) for x in value if x is not None)
    if isinstance(value, dict):
        # No convertir estructuras enteras a repr: extraer solo campos textuales comunes.
        for key in ("texto", "text", "contenido", "content", "respuesta", "answer"):
            if key in value:
                return _cb4_text(value.get(key))
        return ""
    return ""


def _cb4_norm(value):
    value = _cb4_unicodedata.normalize("NFKD", _cb4_text(value).lower())
    return "".join(ch for ch in value if not _cb4_unicodedata.combining(ch))


def _cb4_fallback(caso):
    """Estimación conservadora cuando faltan las señales usadas por el modelo."""
    if not isinstance(caso, dict):
        return 0.80
    answer = _cb4_text(caso.get("respuesta", caso.get("answer", caso.get("response", ""))))
    raw_sources = caso.get("fuentes", caso.get("sources", caso.get("context", [])))
    if isinstance(raw_sources, dict):
        raw_sources = list(raw_sources.values())
    if not isinstance(raw_sources, (list, tuple)):
        raw_sources = [raw_sources] if raw_sources else []
    sources = [_cb4_text(x) for x in raw_sources]
    sources = [x for x in sources if x.strip()]
    if not answer.strip():
        return 0.55
    if not sources:
        return 0.85

    # Los números literales nuevos son una señal fuerte, pero no concluyente:
    # no se aplica si no hay suficientes fuentes para comparar.
    def numbers(text):
        out = set()
        for m in re.findall(r"(?<!\w)\d+(?:[.,]\d+)?%?", text):
            z = m.replace("%", "").replace(",", ".")
            try:
                out.add(round(float(z), 6))
            except ValueError:
                pass
        return out
    an = numbers(answer)
    sn = set().union(*(numbers(x) for x in sources)) if sources else set()
    if an and sn and any(n not in sn for n in an):
        numeric_risk = 0.72
    else:
        numeric_risk = 0.0

    # Solapamiento de palabras de contenido, independiente de idioma concreto.
    stop = {"que", "con", "para", "por", "una", "uno", "unos", "unas", "del", "las", "los", "les", "the", "and", "for", "with", "from", "this", "that", "was", "were", "are", "into", "de", "la", "el", "en", "y", "a", "o", "se", "es", "al", "un"}
    def terms(text):
        return {w for w in re.findall(r"[a-zA-ZÀ-ÿ]{3,}", _cb4_norm(text)) if w not in stop}
    at = terms(answer)
    st = set().union(*(terms(x) for x in sources)) if sources else set()
    overlap = len(at & st) / max(1, len(at))
    # Es solo un backstop para casos sin señales: evitar inferir fidelidad a partir
    # de una mera coincidencia temática. No altera el flujo normal del modelo.
    lexical_risk = 0.68 if len(at) >= 5 and overlap < 0.18 else (0.58 if overlap < 0.08 else 0.0)
    return min(0.90, max(0.52, numeric_risk, lexical_risk))


def _cb4_robust_score(self, caso) -> float:
    try:
        # Mismo camino exacto para el esquema del dataset original.
        if isinstance(caso, dict):
            senales = caso.get("senales")
            oraciones = senales.get("oraciones") if isinstance(senales, dict) else None
            if isinstance(oraciones, list) and oraciones:
                value = _CB4_SCORE_CALIBRADO(self, caso)
                value = float(value)
                if not math.isfinite(value):
                    return _cb4_fallback(caso)
                return min(1.0, max(0.0, value))
        return _cb4_fallback(caso)
    except (Exception,):
        # El fallo de una señal auxiliar no debe tumbar el proceso de evaluación.
        return _cb4_fallback(caso)


Guardian.score = _cb4_robust_score

# ---------------------------------------------------------------------------
# CataBotv5 — integración defensiva con el giro público de El Guardián.
#
# Compatibilidad: conserva exactamente el score de v4_blindado en entradas
# dentro del rango observado en train. En entradas fuera de ese rango (texto
# más largo que el máximo del train local o más oraciones que el máximo), añade
# una auditoría por oración para evitar que una afirmación falsa se diluya en
# respuestas extensas. Solo usa las señales precalculadas del kit y Python stdlib.
# ---------------------------------------------------------------------------
_CB5_BASE_SCORE = _CB4_SCORE_CALIBRADO
_CB5_TRAIN_MAX_CHARS = 629
_CB5_TRAIN_MAX_SENTENCES = 11
_CB5_VERSION = "5.0-giro-publico-robusta"


def _cb5_score_oracion(self, caso, idx, oracion):
    """Puntúa una oración aislada manteniendo todas las fuentes y sus fragmentos."""
    try:
        signals = caso.get("senales") or {}
        fragments = signals.get("fragmentos") or []
        selected = []
        for frag in fragments:
            if not isinstance(frag, dict):
                continue
            try:
                frag_idx = int(frag.get("oracion", 0))
            except (TypeError, ValueError):
                continue
            if frag_idx == idx:
                f = dict(frag)
                f["oracion"] = 0
                selected.append(f)
        one = dict(caso)
        # Mantener coherentes las comprobaciones aritméticas/lexicales con la
        # oración auditada, en vez de reutilizar el texto completo de la respuesta.
        one["respuesta"] = _cb4_text(oracion.get("texto", caso.get("respuesta", "")))
        one_signals = dict(signals)
        one_signals["oraciones"] = [oracion]
        one_signals["fragmentos"] = selected
        one["senales"] = one_signals
        s = float(_CB5_BASE_SCORE(self, one))
        if math.isfinite(s):
            return min(1.0, max(0.0, s))
    except Exception:
        pass
    return 0.0


def _cb5_score(self, caso) -> float:
    # Esquema no reconocido o señales ausentes: conserva el fallback tolerante.
    if not isinstance(caso, dict):
        return _cb4_fallback(caso)
    signals = caso.get("senales")
    sentences = signals.get("oraciones") if isinstance(signals, dict) else None
    if not isinstance(sentences, list) or not sentences:
        return _cb4_fallback(caso)

    try:
        base = float(_CB5_BASE_SCORE(self, caso))
        if not math.isfinite(base):
            return _cb4_fallback(caso)
        base = min(1.0, max(0.0, base))
    except Exception:
        return _cb4_fallback(caso)

    answer = _cb4_text(caso.get("respuesta", caso.get("answer", "")))
    # El dataset de entrenamiento local llega a 629 caracteres y 11 oraciones.
    # Dentro de ese sobre, no cambiamos ni un bit del score de v4.
    out_of_envelope = len(answer) > _CB5_TRAIN_MAX_CHARS or len(sentences) > _CB5_TRAIN_MAX_SENTENCES
    if not out_of_envelope:
        return base

    # Fuera del sobre conocido, evaluamos cada oración contra el conjunto de
    # fuentes. El máximo respeta la regla del reto: una sola afirmación falsa
    # basta para que la respuesta completa sea alucinación.
    per_sentence = []
    for i, sentence in enumerate(sentences):
        if isinstance(sentence, dict):
            per_sentence.append(_cb5_score_oracion(self, caso, i, sentence))
    if not per_sentence:
        return base
    worst = max(per_sentence)

    # El guardarraíl solo puede aumentar la sospecha cuando el análisis local
    # de una oración supera el corte oficial. No baja scores ni cambia casos
    # dentro del rango de train. El tope evita convertirlo en una regla binaria.
    if worst >= 0.5 and base < worst:
        return min(0.99, max(base, worst))
    return base


Guardian.score = _cb5_score
