from ega.types import Unit
from ega.v2.render import SafeAnswerRenderer


def test_safe_answer_renderer_mixed_decisions_are_deterministic() -> None:
    rendered = SafeAnswerRenderer().render(
        units=[
            Unit(id="u1", text="Accepted claim.", metadata={}),
            Unit(id="u2", text="Abstained claim.", metadata={}),
            Unit(id="u3", text="Rejected claim.", metadata={}),
        ],
        decisions={"u1": "accept", "u2": "abstain", "u3": "reject"},
        used_evidence={"u1": ["e3", "e1", "e1"], "u2": ["e2"], "u3": ["e9"]},
    )

    assert rendered.final_text == "Accepted claim. [e1, e3]"
    assert [claim.unit_id for claim in rendered.accepted_claims] == ["u1"]
    assert rendered.accepted_claims[0].citations == ["e1", "e3"]
    assert [claim.unit_id for claim in rendered.abstained_claims] == ["u2"]
    assert rendered.abstained_claims[0].citations == ["e2"]
    assert "Abstained claim." not in rendered.final_text
    assert "Rejected claim." not in rendered.final_text
    assert rendered.summary == {
        "accepted_count": 1,
        "abstained_count": 1,
        "rejected_count": 1,
    }
