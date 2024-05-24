from pydantic import BaseModel


class AddressInput(BaseModel):
    addresses: str
