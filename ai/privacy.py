"""Garde-fou de confidentialité : unique point de construction des données sortantes.

Seules des clés marchandes normalisées (chaînes de caractères) peuvent être
envoyées à l'API pour la catégorisation — jamais de montants, de soldes, de
dates de transaction ni d'identifiants de compte. Toute la couche IA doit
construire son contenu sortant via ce module.
"""

import json
from collections.abc import Sequence


def build_categorization_payload(merchant_keys: Sequence[str]) -> str:
    """Construit le message utilisateur envoyé à l'API de catégorisation.

    N'accepte qu'une séquence de chaînes de caractères ; toute autre valeur
    (nombre, dict, ligne de DataFrame…) lève TypeError. Les montants ne
    peuvent donc pas atteindre l'API par construction.
    """
    if isinstance(merchant_keys, (str, bytes)):
        raise TypeError("merchant_keys doit être une séquence de chaînes, pas une chaîne unique")

    keys = list(merchant_keys)
    for key in keys:
        if not isinstance(key, str):
            raise TypeError(
                "Seules des chaînes marchandes peuvent être envoyées à l'API "
                f"(reçu {type(key).__name__})"
            )
    return json.dumps(keys, ensure_ascii=False)
