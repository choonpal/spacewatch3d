"""Explicit user-authorized server credential storage, outside source/results."""
import getpass
import os
from pathlib import Path
import stat


def location():
    return Path.home() / '.config' / 'spacewatch' / 'credentials' / 'gemini.key'


def api_key():
    if os.environ.get('GEMINI_API_KEY'):
        return os.environ['GEMINI_API_KEY']
    path = location()
    if not path.exists():
        raise RuntimeError('Gemini authentication is not configured')
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or
            stat.S_IMODE(info.st_mode) & 0o077):
        raise RuntimeError('Credential file must be owner-only and not a symlink')
    value = path.read_text().strip()
    if not value:
        raise RuntimeError('Empty Gemini credential')
    return value


def configure():
    """Only invoked explicitly. Secret enters through an echo-off terminal."""
    if not os.isatty(0):
        raise RuntimeError('Use an interactive echo-off terminal for credential setup')
    value = getpass.getpass('Gemini API credential (input hidden): ').strip()
    if not value or '\n' in value or '\r' in value:
        raise ValueError('Nonempty single-line credential required')
    path = location()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    # Never overwrite an existing unrelated credential without a separate choice.
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, 'w') as stream:
        stream.write(value + '\n')
    print('Credential stored outside source/results; owner-only permissions.')


if __name__ == '__main__':
    configure()
