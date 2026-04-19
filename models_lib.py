import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Must be set BEFORE importing torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Version 2.2 - Added Swin Transformer
print("### Loading Models Library v2.2 (CNN / RNN / ViT / Swin) ###")

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(1, channels // reduction), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, channels // reduction), channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x * self.ca(x)
        max_o, _ = torch.max(x, dim=1, keepdim=True)
        avg_o = torch.mean(x, dim=1, keepdim=True)
        return x * self.sa(torch.cat([max_o, avg_o], dim=1))

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.cbam = CBAM(out_ch)

    def forward(self, x):
        return self.cbam(self.net(x))

class MiniUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, features=[16, 32, 64]):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        prev = in_ch
        for f in features:
            self.encoders.append(DoubleConv(prev, f))
            self.pools.append(nn.MaxPool2d(2))
            prev = f

        self.bottleneck = DoubleConv(prev, prev * 2)
        prev = prev * 2

        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(prev, f, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(f * 2, f))
            prev = f

        self.head = nn.Conv2d(prev, out_ch, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for up, dec, skip in zip(self.upconvs, self.decoders, skips):
            x = up(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = dec(torch.cat([skip, x], dim=1))

        return self.head(x)

class CNNClassifier(nn.Module):
    def __init__(self, n_classes=9):
        super().__init__()
        # Matches Notebook 02 Architecture exactly
        self.layer1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.cbam1  = CBAM(32)
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.cbam2  = CBAM(64)
        self.layer3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.cbam3  = CBAM(128)
        self.layer4 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(256, n_classes))

    def forward(self, x):
        x = self.cbam1(self.layer1(x))
        x = self.cbam2(self.layer2(x))
        x = self.cbam3(self.layer3(x))
        x = self.layer4(x)
        return self.head(x)

class BiGRUClassifier(nn.Module):
    def __init__(self, n_classes=9, hidden=128, n_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(80 * 3, hidden)
        self.gru = nn.GRU(hidden, hidden, n_layers, batch_first=True,
                          bidirectional=True, dropout=0.3)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, n_classes))

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H, W * C)
        x = self.input_proj(x)
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])

class TransBlock(nn.Module):
    def __init__(self, dim=128, heads=4, mlp_ratio=3, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, dim*mlp_ratio), nn.GELU(), nn.Dropout(drop),
            nn.Linear(dim*mlp_ratio, dim), nn.Dropout(drop))
    def forward(self, x):
        n = self.norm1(x); x = x + self.attn(n,n,n)[0]
        return x + self.mlp(self.norm2(x))

class ViTClassifier(nn.Module):
    def __init__(self, img_size=80, patch_size=8, in_ch=3, n_classes=9,
                 embed_dim=128, depth=4, heads=4):
        super().__init__()
        n_patches = (img_size // patch_size) ** 2
        self.patch_proj = nn.Conv2d(in_ch, embed_dim, patch_size, patch_size)
        self.cls_token  = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed  = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim) * 0.02)
        self.blocks     = nn.Sequential(*[TransBlock(embed_dim, heads) for _ in range(depth)])
        self.norm       = nn.LayerNorm(embed_dim)
        self.head       = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_proj(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.norm(self.blocks(x))
        return self.head(x[:, 0])

# ─────────────────────────────────────────────────────────────────────────────
# Swin Transformer Classifier (Notebook 04)
# ─────────────────────────────────────────────────────────────────────────────

def _window_partition(x, ws):
    B, H, W, C = x.shape
    x = x.view(B, H//ws, ws, W//ws, ws, C)
    return x.permute(0,1,3,2,4,5).contiguous().view(-1, ws, ws, C)

def _window_reverse(windows, ws, H, W):
    B = int(windows.shape[0] / (H * W / ws / ws))
    x = windows.view(B, H//ws, W//ws, ws, ws, -1)
    return x.permute(0,1,3,2,4,5).contiguous().view(B, H, W, -1)

class _WindowAttn(nn.Module):
    def __init__(self, dim, window_size, num_heads, dropout=0.0):
        super().__init__()
        self.ws = window_size; self.nh = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.rpb = nn.Parameter(torch.zeros((2*window_size-1)**2, num_heads))
        nn.init.trunc_normal_(self.rpb, std=0.02)
        coords = torch.stack(torch.meshgrid(torch.arange(window_size), torch.arange(window_size)))  # 'ij' is default
        cf = torch.flatten(coords, 1)
        rc = cf[:,:,None] - cf[:,None,:]
        rc = rc.permute(1,2,0).contiguous()
        rc[:,:,0] += window_size-1; rc[:,:,1] += window_size-1
        rc[:,:,0] *= 2*window_size-1
        self.register_buffer("rpi", rc.sum(-1))
        self.qkv = nn.Linear(dim, dim*3); self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N,3,self.nh,C//self.nh).permute(2,0,3,1,4)
        q,k,v = qkv.unbind(0)
        attn = (q*self.scale) @ k.transpose(-2,-1)
        bias = self.rpb[self.rpi.view(-1)].view(self.ws**2,self.ws**2,-1).permute(2,0,1).unsqueeze(0)
        attn = attn + bias
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_//nW, nW, self.nh, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.nh, N, N)
        attn = self.drop(F.softmax(attn, dim=-1))
        return self.proj((attn @ v).transpose(1,2).reshape(B_, N, C))

class _SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=4, shift=False, mlp_ratio=4.0, dropout=0.1, resolution=None):
        super().__init__()
        self.ws = window_size; self.ss = window_size//2 if shift else 0
        self.res = resolution
        self.n1 = nn.LayerNorm(dim); self.n2 = nn.LayerNorm(dim)
        self.attn = _WindowAttn(dim, window_size, num_heads, dropout)
        self.mlp  = nn.Sequential(nn.Linear(dim, int(dim*mlp_ratio)), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(int(dim*mlp_ratio), dim), nn.Dropout(dropout))
        if self.ss > 0 and resolution:
            H,W = resolution
            mask = torch.zeros(1,H,W,1)
            cnt=0
            for hs in (slice(0,-window_size), slice(-window_size,-self.ss), slice(-self.ss,None)):
                for ws in (slice(0,-window_size), slice(-window_size,-self.ss), slice(-self.ss,None)):
                    mask[:,hs,ws,:]=cnt; cnt+=1
            mw = _window_partition(mask, window_size).view(-1,window_size**2)
            am = mw.unsqueeze(1)-mw.unsqueeze(2)
            am = am.masked_fill(am!=0,-100.).masked_fill(am==0,0.)
        else:
            am=None
        self.register_buffer("am", am)

    def forward(self, x):
        H,W = self.res; B,L,C = x.shape; res=x
        x = self.n1(x).view(B,H,W,C)
        if self.ss>0: x=torch.roll(x,(-self.ss,-self.ss),(1,2))
        xw = _window_partition(x,self.ws).view(-1,self.ws**2,C)
        xw = self.attn(xw, self.am)
        x = _window_reverse(xw.view(-1,self.ws,self.ws,C),self.ws,H,W)
        if self.ss>0: x=torch.roll(x,(self.ss,self.ss),(1,2))
        x=x.view(B,H*W,C)+res
        return x+self.mlp(self.n2(x))

class _PatchMerging(nn.Module):
    def __init__(self, dim, res):
        super().__init__()
        self.res=res; self.norm=nn.LayerNorm(4*dim); self.red=nn.Linear(4*dim,2*dim,bias=False)
    def forward(self,x):
        H,W=self.res; B,L,C=x.shape; x=x.view(B,H,W,C)
        x=torch.cat([x[:,0::2,0::2],x[:,1::2,0::2],x[:,0::2,1::2],x[:,1::2,1::2]],-1).view(B,-1,4*C)
        return self.red(self.norm(x))

class SwinClassifier(nn.Module):
    """Swin Transformer for 80x80 RBC images, 9 classes."""
    def __init__(self, img_size=80, patch_size=4, in_ch=3, n_classes=9,
                 embed_dim=64, depths=(2,2,2), num_heads=(4,8,16),
                 window_size=5, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.pe = nn.Conv2d(in_ch, embed_dim, patch_size, patch_size)
        self.pn = nn.LayerNorm(embed_dim)
        self.pd = nn.Dropout(dropout)
        res = img_size // patch_size
        self.stages = nn.ModuleList()
        dim = embed_dim
        for i, depth in enumerate(depths):
            blocks = nn.ModuleList([_SwinBlock(dim, num_heads[i], window_size,
                                               shift=(j%2==1), mlp_ratio=mlp_ratio,
                                               dropout=dropout, resolution=(res,res))
                                    for j in range(depth)])
            ds = _PatchMerging(dim,(res,res)) if i<len(depths)-1 else None
            self.stages.append(nn.ModuleList([blocks, ds or nn.Identity()]))
            if i<len(depths)-1: dim*=2; res//=2
        self.norm = nn.LayerNorm(dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(dim, n_classes))

    def forward(self, x):
        B=x.shape[0]; x=self.pe(x); C,H,W=x.shape[1],x.shape[2],x.shape[3]
        x=self.pd(self.pn(x.flatten(2).transpose(1,2)))
        for blocks, ds in self.stages:
            for blk in blocks: x=blk(x)
            if not isinstance(ds, nn.Identity): x=ds(x)
        return self.head(self.pool(self.norm(x).transpose(1,2)).squeeze(-1))
