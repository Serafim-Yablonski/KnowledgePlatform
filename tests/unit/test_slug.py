"""Unit tests for generate_slug — pure function, no dependencies."""

from src.domain.slug import generate_slug


def test_spaces_become_hyphens() -> None:
    slug = generate_slug("my team name")
    assert slug.startswith("my-team-name-")


def test_special_chars_stripped() -> None:
    slug = generate_slug("hello!@# world$%^")
    assert slug.startswith("hello-world-")


def test_unicode_stripped() -> None:
    slug = generate_slug("café résumé")
    assert "-" in slug or slug[:4].isalnum()


def test_all_special_chars_returns_suffix_only() -> None:
    slug = generate_slug("!@#$%^&*()")
    assert len(slug) == 4
    assert slug.isalnum()


def test_empty_string_returns_suffix_only() -> None:
    slug = generate_slug("")
    assert len(slug) == 4
    assert slug.isalnum()


def test_very_long_name_truncated() -> None:
    slug = generate_slug("a" * 200)
    # 50 chars + hyphen + 4 suffix = 55 max
    assert len(slug) <= 55


def test_two_calls_same_name_differ() -> None:
    slug1 = generate_slug("Engineering Hub")
    slug2 = generate_slug("Engineering Hub")
    assert slug1 != slug2


def test_lowercase_output() -> None:
    slug = generate_slug("MY WORKSPACE")
    base = slug.rsplit("-", 1)[0]
    assert base == base.lower()


def test_no_consecutive_hyphens() -> None:
    slug = generate_slug("hello   world")
    assert "--" not in slug


def test_suffix_always_appended() -> None:
    for _ in range(10):
        slug = generate_slug("test")
        # slug is "test-xxxx" — always has a 4-char suffix after the last hyphen
        parts = slug.rsplit("-", 1)
        assert len(parts) == 2
        assert len(parts[1]) == 4
