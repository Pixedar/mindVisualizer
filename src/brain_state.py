"""Brain region state database.

Stores the current cognitive/functional state of each brain region. States can be:
  - Auto-initialized from a global brain state description via GPT
  - Manually edited per-region
  - Cleared and re-initialized at any time
  - Persisted to disk as JSON

Used by the probe system to simulate information propagation: when a probe
enters a region, the state can be modified and propagated to downstream regions.
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path
from typing import Optional


def _ensure_ssl():
    """Fix SSL cert path for environments where Git overrides it.

    Must run BEFORE importing openai/langchain so the httpx client
    picks up the correct certificate.
    """
    try:
        import certifi
        cert_file = certifi.where()
        cur = os.environ.get("SSL_CERT_FILE", "")
        if not cur or not os.path.isfile(cur):
            os.environ["SSL_CERT_FILE"] = cert_file
        cur2 = os.environ.get("REQUESTS_CA_BUNDLE", "")
        if not cur2 or not os.path.isfile(cur2):
            os.environ["REQUESTS_CA_BUNDLE"] = cert_file
    except ImportError:
        pass


# Fix SSL before any network library is imported
_ensure_ssl()

import openai
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


DEFAULT_STATE_FILE = Path(__file__).resolve().parent.parent / "data" / "brain_states.json"


DEFAULT_MODEL = "gpt-5.4-mini"
HQ_MODEL = "gpt-5.4"


class BrainStateDB:
    """Database of brain region states, persisted to JSON."""

    def __init__(self, path: Path | str = DEFAULT_STATE_FILE, model: str = DEFAULT_MODEL,
                 debug: bool = False):
        self.path = Path(path)
        self.model = model
        self.debug = debug
        self.states: dict[str, str] = {}  # region_name -> state description
        self.global_state: str = ""  # the overall brain state description
        if self.path.exists():
            self.load()

    def _debug_prompt(self, method_name: str, prompt_text: str):
        """Print the full prompt if debug mode is enabled."""
        if self.debug:
            print(f"\n{'='*60}")
            print(f"[DEBUG PROMPT] {method_name}")
            print(f"{'='*60}")
            print(prompt_text)
            print(f"{'='*60}\n")
            sys.stdout.flush()

    # ---------- persistence ----------

    def load(self):
        """Load states from JSON file."""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.states = data.get("states", {})
            self.global_state = data.get("global_state", "")
            print(f"[brain-state] loaded {len(self.states)} region states")
        except Exception as e:
            print(f"[brain-state] could not load {self.path}: {e}")

    def save(self):
        """Save states to JSON file."""
        data = {
            "global_state": self.global_state,
            "states": self.states,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        print(f"[brain-state] saved {len(self.states)} region states to {self.path}")

    def clear(self):
        """Clear all states."""
        self.states.clear()
        self.global_state = ""
        if self.path.exists():
            self.path.unlink()
        print("[brain-state] cleared all states")

    # ---------- access ----------

    def get(self, region_name: str) -> str:
        return self.states.get(region_name, "")

    def set(self, region_name: str, state: str):
        self.states[region_name] = state

    def has_states(self) -> bool:
        return bool(self.states)

    # ---------- GPT initialization ----------

    def initialize_from_global(self, global_state: str, region_names: list[str],
                               callback=None) -> str:
        """Use GPT to generate states for all regions from a global brain state.

        Args:
            global_state: e.g. "someone thinking about loved ones" or "" for GPT to choose
            region_names: list of region names to generate states for
            callback: optional fn(status_msg) for progress updates

        Returns:
            Summary of initialization.
        """
        _ensure_ssl()

        load_dotenv()


        if not global_state:
            global_state = "a resting state with spontaneous mind-wandering"

        self.global_state = global_state

        # Build region list (batch into groups of ~30 for efficiency)
        all_states = {}
        batch_size = 30

        for i in range(0, len(region_names), batch_size):
            batch = region_names[i:i + batch_size]
            if callback:
                callback(f"Generating states for regions {i+1}-{i+len(batch)}/{len(region_names)}...")

            region_list = "\n".join(f"- {name}" for name in batch)

            prompt = ChatPromptTemplate.from_template(
                """You are a neuroscience expert. The brain is currently in this overall state:
"{global_state}"

For each brain region below, write a SHORT (1-2 sentences) specific description of
what this region is likely doing right now given the overall brain state.

IMPORTANT RULES:
- Be SPECIFIC about the actual cognitive content, not generic function descriptions
- Don't force every region to match the global state — some regions may be doing
  their own thing (e.g., sensory processing, homeostasis) independent of the global state
- First consider what the region generally does, then derive what it's specifically
  doing in this context
- Focus on CURRENT ACTIVITY, not general capabilities

Regions:
{regions}

Reply as JSON object mapping region name to state description. Example:
{{"Amygdala (AMY)": "Low-level monitoring for threats; no active fear processing", ...}}
Only output valid JSON, nothing else."""
            )

            llm = ChatOpenAI(model=self.model, temperature=0.4, max_tokens=4000)
            chain = prompt | llm | StrOutputParser()

            self._debug_prompt("initialize_from_global",
                               prompt.format(global_state=global_state, regions=region_list))

            try:
                result = chain.invoke({
                    "global_state": global_state,
                    "regions": region_list,
                })
                # Parse JSON from response
                result = result.strip()
                if result.startswith("```"):
                    result = result.split("\n", 1)[1].rsplit("```", 1)[0]
                parsed = json.loads(result)
                all_states.update(parsed)
                # Show each region's state in real time
                for rname, rstate in parsed.items():
                    if callback:
                        callback(f"  {rname}: {rstate}")
            except Exception as e:
                print(f"[brain-state] GPT batch error: {e}")
                for name in batch:
                    all_states[name] = f"(initialization failed: {e})"

        self.states = all_states
        self.save()
        return f"Initialized {len(self.states)} region states for: {global_state}"

    def validate_perturbation(self, region_name: str, perturbation: str) -> dict:
        """Check if a perturbation is appropriate for a brain region.

        Returns:
            dict with keys:
              - valid (bool): True if perturbation is plausible
              - region_function (str): what this region does
              - suggestion (str): suggested alternative if not valid
              - warning (str): warning message if perturbation is questionable
        """
        _ensure_ssl()

        load_dotenv()


        current = self.states.get(region_name, "unknown state")

        prompt = ChatPromptTemplate.from_template(
            """You are a neuroscience expert. Evaluate whether this perturbation makes sense
for the specified brain region.

Region: {region}
Current state: {current}
Requested perturbation: "{perturbation}"

Consider:
1. What does this brain region actually do? (its primary functions)
2. Is the requested perturbation something this region CAN process?
3. If not, what WOULD be an appropriate perturbation for this region?

Reply as JSON:
{{
  "valid": true/false,
  "region_function": "brief description of what this region does",
  "warning": "warning if questionable (empty string if fine)",
  "suggestion": "suggested alternative perturbation if invalid (empty string if valid)"
}}
Only output valid JSON."""
        )

        llm = ChatOpenAI(model=self.model, temperature=0.2, max_tokens=300)
        chain = prompt | llm | StrOutputParser()

        self._debug_prompt("validate_perturbation",
                           prompt.format(region=region_name, current=current,
                                         perturbation=perturbation))

        try:
            result = chain.invoke({
                "region": region_name,
                "current": current,
                "perturbation": perturbation,
            }).strip()
            if result.startswith("```"):
                result = result.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(result)
        except Exception as e:
            return {"valid": True, "region_function": "unknown",
                    "warning": f"(validation failed: {e})", "suggestion": ""}

    def propose_perturbations(self, region_name: str) -> list[str]:
        """Ask GPT to propose 4 plausible perturbations for a brain region.

        Uses the openai SDK directly (not langchain) for reliability in
        background threads.

        Args:
            region_name: the region to perturb

        Returns:
            List of 4 perturbation description strings.
        """
        import sys
        _defaults = [
            "Heightened activity in this region",
            "Suppressed activity in this region",
            "Shift to an alternative processing mode",
            "Disrupted connectivity with downstream regions",
        ]

        _ensure_ssl()

        load_dotenv()

        current = self.states.get(region_name, "unknown state")

        # Use a shorter name for prompt if the full name is very long
        short_name = region_name
        if len(short_name) > 80 and "(" in short_name:
            short_name = short_name.split("(", 1)[1].rstrip(")")

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("[brain-state] OPENAI_API_KEY not set, using defaults")
            sys.stdout.flush()
            return _defaults

        print(f"[brain-state] propose_perturbations: calling OpenAI for '{short_name}'...")
        sys.stdout.flush()

        try:
            client = openai.OpenAI(api_key=api_key, timeout=30.0)

            msg_content = (
                f"You are a neuroscience expert. A user wants to perturb a brain region "
                f"in a resting-state simulation.\n\n"
                f"Region: {short_name}\n"
                f"Current state: \"{current}\"\n\n"
                f"Based on the region's known functions and its current state, propose "
                f"exactly 4 different plausible ways this region's state could change. "
                f"Each should be specific to this region's actual function, a realistic "
                f"state change, described in 1 short sentence, and diverse from each other.\n\n"
                f"Reply as a JSON array of exactly 4 strings:\n"
                f'["perturbation 1", "perturbation 2", "perturbation 3", "perturbation 4"]\n'
                f"Only output valid JSON."
            )

            self._debug_prompt("propose_perturbations", msg_content)

            print(f"[brain-state] propose_perturbations: sending API request...")
            sys.stdout.flush()

            def _call():
                return client.chat.completions.create(
                    model=self.model,
                    temperature=0.5,
                    max_completion_tokens=400,
                    messages=[{"role": "user", "content": msg_content}],
                )
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_call)
                try:
                    response = future.result(timeout=30)
                except FutureTimeout:
                    print("[brain-state] propose_perturbations: TIMEOUT after 30s")
                    sys.stdout.flush()
                    return _defaults

            result = response.choices[0].message.content.strip()
            print(f"[brain-state] propose_perturbations: got response, parsing...")
            sys.stdout.flush()

            if result.startswith("```"):
                result = result.split("\n", 1)[1].rsplit("```", 1)[0]
            proposals = json.loads(result)
            if isinstance(proposals, list) and len(proposals) >= 4:
                return proposals[:4]
            return proposals if isinstance(proposals, list) else _defaults
        except Exception as e:
            print(f"[brain-state] proposal error: {type(e).__name__}: {e}")
            sys.stdout.flush()
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            return _defaults

    def alter_region_state(self, region_name: str, modification: str,
                           callback=None, skip_validation=False) -> str:
        """Use GPT to alter a specific region's state.

        Args:
            region_name: the region to modify
            modification: description of how to change it
            callback: optional fn(status_msg) for progress
            skip_validation: skip perturbation validation

        Returns:
            The new state description.
        """
        _ensure_ssl()

        load_dotenv()


        # Validate perturbation first
        if not skip_validation:
            if callback:
                callback(f"Validating perturbation for {region_name}...")
            validation = self.validate_perturbation(region_name, modification)

            if callback:
                callback(f"Region function: {validation.get('region_function', '?')}")

            if not validation.get("valid", True):
                warning = validation.get("warning", "")
                suggestion = validation.get("suggestion", "")
                msg = f"[WARNING] Perturbation may not be appropriate for {region_name}."
                if warning:
                    msg += f"\n  Reason: {warning}"
                if suggestion:
                    msg += f"\n  Suggestion: {suggestion}"
                if callback:
                    callback(msg)
                print(msg)
                # Still proceed but note the warning
            elif validation.get("warning"):
                if callback:
                    callback(f"Note: {validation['warning']}")

        current = self.states.get(region_name, "unknown state")

        prompt = ChatPromptTemplate.from_template(
            """You are a neuroscience expert. A brain region's state needs to be modified.

Region: {region}
Current state: {current}
Modification requested: {modification}

Write a new SHORT (1-2 sentences) specific state description that incorporates
the requested modification while staying neuroscientifically plausible.
Only output the new state description, nothing else."""
        )

        llm = ChatOpenAI(model=self.model, temperature=0.3, max_tokens=200)
        chain = prompt | llm | StrOutputParser()

        self._debug_prompt("alter_region_state",
                           prompt.format(region=region_name, current=current,
                                         modification=modification))

        try:
            new_state = chain.invoke({
                "region": region_name,
                "current": current,
                "modification": modification,
            }).strip()
            self.states[region_name] = new_state
            self.save()
            return new_state
        except Exception as e:
            return f"(alter failed: {e})"

    def propagate_through_graph(self, source_region: str,
                                connections: list[dict],
                                A,  # numpy ndarray (R x R)
                                roi_names: list[str],
                                callback=None) -> dict[str, str]:
        """Propagate perturbation through the connectivity graph, depth by depth.

        For each affected region individually:
          1. Determine if the incoming signal is strong or weak (relative to that
             region's other connections)
          2. Find additional un-perturbed regions strongly connected to this target
             for context
          3. Ask GPT: given region X's function, its previous state, the incoming
             signal from Y (strong/weak), and contextual connections — what does
             X's state change to?

        Processes depth-1 targets first, then depth-2 using already-updated states,
        so the signal cascades realistically.

        Args:
            source_region: name of the perturbed region
            connections: list of dicts from get_strongest_connections, each with
                         target_idx, weight, abs_weight, depth, source_idx
            A: full connectivity matrix (R x R numpy array)
            roi_names: list of ROI names matching A's indices
            callback: optional fn(status_msg)

        Returns:
            Dict of {region_name: new_state} for all updated regions.
        """
        import numpy as np
        _ensure_ssl()

        load_dotenv()


        llm = ChatOpenAI(model=self.model, temperature=0.3, max_tokens=300)

        # Group connections by depth
        by_depth: dict[int, list[dict]] = {}
        for c in connections:
            by_depth.setdefault(c["depth"], []).append(c)

        # Build name->index map
        name_to_idx = {n: i for i, n in enumerate(roi_names)}

        # Track before/after states and which regions were updated
        before_states: dict[str, str] = {}
        all_updates: dict[str, str] = {}
        perturbed_idxs = set()
        source_idx = name_to_idx.get(source_region)
        if source_idx is not None:
            perturbed_idxs.add(source_idx)

        # Diagonal-zeroed matrix for connection lookups
        Ac = np.array(A, dtype=np.float32, copy=True)
        np.fill_diagonal(Ac, 0.0)

        # Process depth by depth
        max_depth = max(by_depth.keys()) if by_depth else 0
        for d in range(1, max_depth + 1):
            depth_conns = by_depth.get(d, [])
            if not depth_conns:
                continue

            if callback:
                callback(f"Propagating depth {d}: {len(depth_conns)} regions...")

            for c in depth_conns:
                ti = c["target_idx"]
                target_name = roi_names[ti]
                from_idx = c["source_idx"]
                from_name = roi_names[from_idx]

                # Skip if target has no state
                target_prev_state = self.states.get(target_name, "")
                if not target_prev_state:
                    target_prev_state = "unknown / not initialized"
                before_states[target_name] = target_prev_state

                # Get the incoming source's current state (may have been
                # updated in a previous depth iteration)
                from_state = all_updates.get(from_name,
                                              self.states.get(from_name, "unknown"))

                # ---- Determine signal strength and type ----
                # Compare this connection weight to the target's other incoming
                # connections (row ti of A)
                incoming_weights = np.abs(Ac[ti, :])
                incoming_weights[ti] = 0  # no self
                median_incoming = float(np.median(incoming_weights[incoming_weights > 0])) \
                    if np.any(incoming_weights > 0) else 0.001
                conn_abs_w = c["abs_weight"]
                if conn_abs_w > median_incoming * 2.0:
                    strength_label = "STRONG (well above average)"
                elif conn_abs_w > median_incoming * 0.8:
                    strength_label = "moderate"
                else:
                    strength_label = "weak (below average)"

                # Determine excitatory vs inhibitory based on connection sign
                raw_weight = c.get("weight", conn_abs_w)
                if raw_weight < 0:
                    sign_label = "INHIBITORY (negative connection weight)"
                else:
                    sign_label = "EXCITATORY (positive connection weight)"

                # ---- Find context: other strong un-perturbed connections ----
                context_lines = []
                top_others = np.argsort(incoming_weights)[::-1][:10]
                for oi in top_others:
                    if int(oi) == from_idx or int(oi) in perturbed_idxs:
                        continue
                    oi = int(oi)
                    other_name = roi_names[oi]
                    other_state = self.states.get(other_name, "")
                    if not other_state:
                        continue
                    other_w = float(incoming_weights[oi])
                    if other_w < conn_abs_w * 0.3:
                        break  # only include meaningfully strong ones
                    context_lines.append(
                        f"- {other_name} (connection weight {other_w:.4f}): "
                        f"state = \"{other_state}\""
                    )
                    if len(context_lines) >= 3:
                        break

                context_text = ""
                if context_lines:
                    context_text = (
                        "\n\nAdditional context — other strong connections to this region "
                        "(NOT perturbed, their states remain stable):\n"
                        + "\n".join(context_lines)
                    )

                # ---- GPT call for this single region ----
                prompt = ChatPromptTemplate.from_template(
                    """You are a computational neuroscientist analyzing intrinsic information flow in a resting-state brain network (rs-fMRI effective connectivity).

TARGET REGION: {target}
TARGET PREVIOUS STATE: {target_state}

INCOMING SIGNAL FROM: {source}
INCOMING SIGNAL STRENGTH: {strength} (connection weight: {weight:.4f})
CONNECTION TYPE: {sign}
SOURCE REGION'S CURRENT STATE: "{source_state}"
{context}
CRITICAL INSTRUCTIONS:
1. NO SEMANTIC ECHOING: Do not simply copy the semantic concept of the source region. You must TRANSLATE the incoming signal into the strict anatomical and functional domain of the TARGET REGION. If the source is about "visual beauty", the motor cortex should NOT start "appreciating beauty" — it should show changes in motor readiness or postural tone.
2. RESTING-STATE CONTEXT: The connectivity data reflects intrinsic resting-state dynamics. Information flow here represents spontaneous internal cognition or modulation of the target region's resting equilibrium.
3. INHIBITORY vs EXCITATORY: If the connection is INHIBITORY, the incoming signal SUPPRESSES or DAMPENS the target region's activity. If EXCITATORY, it AMPLIFIES or FACILITATES the target's function. This fundamentally changes the nature of the state change.
4. INTRINSIC DYNAMICS: Focus on how the target region's OWN function shifts, not on relaying the source's content.
5. Output ONLY the precise description of the target region's new state (1-2 sentences). No conversational filler."""
                )

                chain = prompt | llm | StrOutputParser()

                self._debug_prompt("propagate_through_graph",
                                   prompt.format(target=target_name,
                                                 target_state=target_prev_state,
                                                 strength=strength_label,
                                                 sign=sign_label,
                                                 source=from_name,
                                                 source_state=from_state,
                                                 weight=c["abs_weight"],
                                                 context=context_text))

                try:
                    new_state = chain.invoke({
                        "target": target_name,
                        "target_state": target_prev_state,
                        "strength": strength_label,
                        "sign": sign_label,
                        "source": from_name,
                        "source_state": from_state,
                        "weight": c["abs_weight"],
                        "context": context_text,
                    }).strip()
                    all_updates[target_name] = new_state
                    perturbed_idxs.add(ti)

                    if callback:
                        callback(f"  {target_name}: {new_state}")

                except Exception as e:
                    print(f"[brain-state] propagation error for {target_name}: {e}")

        # Record source before state too
        before_states[source_region] = self.states.get(source_region, "unknown")

        # Apply all updates
        for name, new_state in all_updates.items():
            self.states[name] = new_state
        self.save()

        # Attach before_states so summarize_changes can use them
        self._last_before_states = before_states

        return all_updates

    def propagate_through_regions(self, source_region: str, affected_regions: list[str],
                                  flow_strengths: dict[str, float] | None = None,
                                  callback=None) -> dict[str, str]:
        """Simpler propagation for MDN flow mode (no connectivity matrix).

        Uses flow strengths as a proxy for connection weights. Processes regions
        in order of the flow path (which is already serial).
        """
        _ensure_ssl()

        load_dotenv()


        llm = ChatOpenAI(model=self.model, temperature=0.3, max_tokens=300)

        source_state = self.states.get(source_region, "unknown")
        all_updates = {}
        before_states = {source_region: source_state}

        # Process regions in flow order (serial propagation)
        prev_name = source_region
        prev_state = source_state

        if flow_strengths and affected_regions:
            all_strengths = [flow_strengths.get(n, 0.0) for n in affected_regions]
            max_s = max(all_strengths) if all_strengths else 1.0
        else:
            max_s = 1.0

        for name in affected_regions:
            target_state = self.states.get(name, "unknown")
            before_states[name] = target_state

            strength = flow_strengths.get(name, 0.0) if flow_strengths else 0.0
            if max_s > 0:
                rel_strength = strength / max_s
                if rel_strength > 0.6:
                    strength_label = "STRONG"
                elif rel_strength > 0.3:
                    strength_label = "moderate"
                else:
                    strength_label = "weak"
            else:
                strength_label = "moderate"

            prompt = ChatPromptTemplate.from_template(
                """You are a computational neuroscientist analyzing intrinsic information flow in a resting-state brain network.

TARGET REGION: {target}
TARGET PREVIOUS STATE: {target_state}

INCOMING SIGNAL FROM: {source}
INCOMING SIGNAL STRENGTH: {strength}
SOURCE REGION'S CURRENT STATE: "{source_state}"

CRITICAL INSTRUCTIONS:
1. NO SEMANTIC ECHOING: Do not simply copy the semantic concept of the source region. You must TRANSLATE the incoming signal into the strict anatomical and functional domain of the TARGET REGION ({target}). For example, if the source is a visual area processing "edge detection" and the target is a motor area, do NOT say the motor area is doing "edge detection" — describe how the motor area's OWN function shifts in response.
2. RESTING-STATE CONTEXT: This is intrinsic resting-state dynamics, not task-driven activity. Describe subtle modulations, not dramatic activations.
3. INTRINSIC DYNAMICS: Focus on how {target}'s OWN function shifts given the incoming signal. The target region does what IT does, influenced by the source — not what the source does.

Output ONLY the precise description of {target}'s new state (1-2 sentences). No labels, no prefixes."""
            )

            chain = prompt | llm | StrOutputParser()

            self._debug_prompt("propagate_through_regions",
                               prompt.format(target=name, target_state=target_state,
                                             strength=strength_label, source=prev_name,
                                             source_state=prev_state))

            try:
                new_state = chain.invoke({
                    "target": name,
                    "target_state": target_state,
                    "strength": strength_label,
                    "source": prev_name,
                    "source_state": prev_state,
                }).strip()
                all_updates[name] = new_state
                if callback:
                    callback(f"  {name}: {new_state}")
                # Next hop uses this updated state as source
                prev_name = name
                prev_state = new_state
            except Exception as e:
                print(f"[brain-state] propagation error for {name}: {e}")

        # Apply
        for name, new_state in all_updates.items():
            self.states[name] = new_state
        self.save()

        self._last_before_states = before_states
        return all_updates

    def summarize_changes(self, updates: dict[str, str],
                          source_region: str) -> str:
        """Summarize what changed: show before/after states (without revealing the
        perturbation) and ask GPT to build a coherent picture.

        Args:
            updates: dict of {region_name: new_state}
            source_region: the region that was perturbed

        Returns:
            Human-readable coherent summary.
        """
        _ensure_ssl()

        load_dotenv()


        # Get before states (saved by propagation methods)
        before = getattr(self, '_last_before_states', {})

        # Build separate before and after state maps
        all_regions = [source_region] + [n for n in updates if n != source_region]
        before_lines = []
        after_lines = []
        for name in all_regions:
            prev = before.get(name, "unknown")
            current = self.states.get(name, updates.get(name, "unknown"))
            before_lines.append(f"- {name}: \"{prev}\"")
            after_lines.append(f"- {name}: \"{current}\"")

        llm = ChatOpenAI(model=self.model, temperature=0.3, max_tokens=800)

        prompt = ChatPromptTemplate.from_template(
            """You are a network neuroscientist analyzing a macroscopic shift in resting-state brain activity based on effective connectivity changes.

INITIAL BRAIN STATE MAP:
{comparisons_before}

POST-PROPAGATION BRAIN STATE MAP:
{comparisons_after}

CRITICAL INSTRUCTIONS:
1. DO NOT list the regions or compare them one by one.
2. This is a resting-state brain network — interpret changes as shifts in intrinsic functional organization, not external stimulus-response narratives.
3. FOCUS ON INTRINSIC STATES: Synthesize this data into ONE coherent paragraph explaining the overall shift in the subject's internal cognitive, emotional, or physiological baseline.
4. NETWORK LEVEL INTEGRATION: Identify the broad functional domains driving the new equilibrium and describe the holistic network-level transition based purely on the provided state changes.

Provide your coherent resting-state network insight below:"""
        )

        chain = prompt | llm | StrOutputParser()

        self._debug_prompt("summarize_changes",
                           prompt.format(comparisons_before="\n".join(before_lines),
                                         comparisons_after="\n".join(after_lines)))

        try:
            return chain.invoke({
                "comparisons_before": "\n".join(before_lines),
                "comparisons_after": "\n".join(after_lines),
            }).strip()
        except Exception as e:
            return f"(summary failed: {e})"

    def generate_flow_story(self, updates: dict[str, str],
                            source_region: str,
                            connections: list[dict] | None = None) -> str:
        """Generate a narrative story of how information flowed through the brain.

        Unlike summarize_changes (which gives a holistic snapshot), this tells
        the story of the signal's journey: where it started, how each region
        processed and transformed it, and what the downstream effects were.

        Args:
            updates: dict of {region_name: new_state}
            source_region: the origin region
            connections: optional list of connection dicts (with depth, weight info)

        Returns:
            A narrative paragraph describing the information flow journey.
        """
        _ensure_ssl()

        load_dotenv()


        before = getattr(self, '_last_before_states', {})

        # Build ordered flow description
        flow_steps = []
        source_before = before.get(source_region, "unknown")
        source_after = self.states.get(source_region,
                                        updates.get(source_region, "unknown"))
        flow_steps.append(
            f"ORIGIN — {source_region}: \"{source_before}\" -> \"{source_after}\""
        )

        # Order by depth if connections available
        if connections:
            by_depth: dict[int, list] = {}
            for c in connections:
                name = c.get("target_name", "")
                if not name and "target_idx" in c:
                    continue
                by_depth.setdefault(c.get("depth", 1), []).append(c)
            for d in sorted(by_depth.keys()):
                for c in by_depth[d]:
                    name = c.get("target_name", "")
                    if name in updates:
                        prev = before.get(name, "unknown")
                        sign = "excitatory" if c.get("weight", 0) >= 0 else "inhibitory"
                        flow_steps.append(
                            f"DEPTH {d} ({sign}) — {name}: \"{prev}\" -> \"{updates[name]}\""
                        )
        else:
            for name, new_state in updates.items():
                if name == source_region:
                    continue
                prev = before.get(name, "unknown")
                flow_steps.append(f"STEP — {name}: \"{prev}\" -> \"{new_state}\"")

        flow_text = "\n".join(flow_steps)

        llm = ChatOpenAI(model=self.model, temperature=0.4, max_tokens=800)

        prompt = ChatPromptTemplate.from_template(
            """You are a science writer narrating how a signal traveled through a resting-state brain network.

SIGNAL FLOW PATH (in order of propagation):
{flow_path}

Write a SHORT narrative story (2-3 paragraphs) of how the information traveled through the brain:
- Start with where the signal originated and what it carried
- Describe how each region it reached processed and TRANSFORMED the signal according to its own function
- Highlight how the signal's meaning changed as it moved through different functional domains
- End with the overall effect on the brain's resting state

RULES:
- Do NOT list regions mechanically — weave them into a flowing narrative
- Use concrete, vivid language about what each region actually does
- Show how the signal was transformed at each hop, not just passed along
- Keep it grounded in neuroscience but accessible to a general audience"""
        )

        chain = prompt | llm | StrOutputParser()

        self._debug_prompt("generate_flow_story",
                           prompt.format(flow_path=flow_text))

        try:
            return chain.invoke({"flow_path": flow_text}).strip()
        except Exception as e:
            return f"(story generation failed: {e})"
