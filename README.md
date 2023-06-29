## Installation instructions

```
# Install dreal
curl -fsSL https://raw.githubusercontent.com/dreal/dreal4/master/setup/ubuntu/22.04/install.sh | sudo bash
# Test
DREAL_VERSION=4.21.06.2
/opt/dreal/${DREAL_VERSION}/bin/dreal

python3 -m venv venv
source venv/bin/activate
pip install dreal numpy tqdm
```