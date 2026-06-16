
Add-Type -AssemblyName System.Drawing
$ErrorActionPreference = "Stop"
$dir = "C:\\Users\\wb.zhoushujie\\PyCharmMiscProject\\results"
[System.IO.Directory]::CreateDirectory($dir) | Out-Null

function Draw-CenteredText($g, $text, $font, $brush, $rect) {
    $sf = New-Object System.Drawing.StringFormat
    $sf.Alignment = [System.Drawing.StringAlignment]::Center
    $sf.LineAlignment = [System.Drawing.StringAlignment]::Center
    $sf.Trimming = [System.Drawing.StringTrimming]::Word
    $sf.FormatFlags = 0
    $g.DrawString($text, $font, $brush, $rect, $sf)
    $sf.Dispose()
}

function Draw-Box($g, $x, $y, $w, $h, $text, $fill, $pen, $font, $brush) {
    $rect = New-Object System.Drawing.RectangleF($x, $y, $w, $h)
    $path = New-Object System.Drawing.Drawing2D.GraphicsPath
    $r = 18
    $path.AddArc($x, $y, $r, $r, 180, 90)
    $path.AddArc($x + $w - $r, $y, $r, $r, 270, 90)
    $path.AddArc($x + $w - $r, $y + $h - $r, $r, $r, 0, 90)
    $path.AddArc($x, $y + $h - $r, $r, $r, 90, 90)
    $path.CloseFigure()
    $g.FillPath($fill, $path)
    $g.DrawPath($pen, $path)
    Draw-CenteredText $g $text $font $brush $rect
    $path.Dispose()
}

function Draw-Arrow($g, $x1, $y1, $x2, $y2, $pen) {
    $g.DrawLine($pen, $x1, $y1, $x2, $y2)
    $angle = [Math]::Atan2($y2 - $y1, $x2 - $x1)
    $len = 13
    $a1 = $angle + [Math]::PI * 0.82
    $a2 = $angle - [Math]::PI * 0.82
    $p1 = New-Object System.Drawing.PointF(($x2 + $len * [Math]::Cos($a1)), ($y2 + $len * [Math]::Sin($a1)))
    $p2 = New-Object System.Drawing.PointF(($x2 + $len * [Math]::Cos($a2)), ($y2 + $len * [Math]::Sin($a2)))
    $pts = @((New-Object System.Drawing.PointF($x2,$y2)), $p1, $p2)
    $g.FillPolygon($pen.Brush, $pts)
}

function New-Canvas($title, $path) {
    $bmp = New-Object System.Drawing.Bitmap(1600, 1000)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.Clear([System.Drawing.Color]::White)
    $titleFont = New-Object System.Drawing.Font("Microsoft YaHei", 26, [System.Drawing.FontStyle]::Bold)
    $black = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(30,30,30))
    Draw-CenteredText $g $title $titleFont $black (New-Object System.Drawing.RectangleF(70, 25, 1460, 70))
    return @($bmp, $g)
}

function Save-Canvas($bmp, $g, $path) {
    $g.Dispose()
    $bmp.Save($path, [System.Drawing.Imaging.ImageFormat]::Png)
    $bmp.Dispose()
}

$font = New-Object System.Drawing.Font("Microsoft YaHei", 18, [System.Drawing.FontStyle]::Regular)
$smallFont = New-Object System.Drawing.Font("Microsoft YaHei", 15, [System.Drawing.FontStyle]::Regular)
$boldFont = New-Object System.Drawing.Font("Microsoft YaHei", 18, [System.Drawing.FontStyle]::Bold)
$brushText = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(30,30,30))
$blue = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(226,240,255))
$green = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(230,247,232))
$orange = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255,241,220))
$purple = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(242,232,255))
$gray = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(245,246,248))
$penBlue = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(52,113,185), 3)
$penGreen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(69,150,83), 3)
$penOrange = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(210,130,45), 3)
$penPurple = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(120,84,180), 3)
$arrowPen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(70,70,70), 4)
$arrowPen.EndCap = [System.Drawing.Drawing2D.LineCap]::Round
$arrowPen.StartCap = [System.Drawing.Drawing2D.LineCap]::Round

# Fig 1
$tmp = New-Canvas "整体方法流程图" (Join-Path $dir "patent_fig1_overall_flow.png"); $bmp=$tmp[0]; $g=$tmp[1]
$xs = 540; $w = 520; $h = 72; $ys = @(130,240,350,460,570,680,790)
$texts = @("SGCC 智能电表原始数据", "数据清洗与预处理", "多源用电行为特征工程", "随机矩阵理论 RMT 信号增强", "双路径 Transformer 深度表征提取", "PCA 表征压缩与特征融合", "异常用电风险分数与检测结果")
for($i=0; $i -lt $ys.Count; $i++){ Draw-Box $g $xs $ys[$i] $w $h $texts[$i] $blue $penBlue $font $brushText; if($i -lt $ys.Count-1){ Draw-Arrow $g 800 ($ys[$i]+$h) 800 $ys[$i+1] $arrowPen } }
Draw-Box $g 175 680 280 72 "CatBoost" $green $penGreen $font $brushText
Draw-Box $g 175 780 280 72 "XGBoost" $green $penGreen $font $brushText
Draw-Box $g 1145 730 280 72 "LightGBM" $green $penGreen $font $brushText
Draw-Arrow $g 540 716 455 716 $arrowPen; Draw-Arrow $g 455 816 540 816 $arrowPen; Draw-Arrow $g 1060 716 1145 766 $arrowPen; Draw-Arrow $g 1145 766 1060 826 $arrowPen
Save-Canvas $bmp $g (Join-Path $dir "patent_fig1_overall_flow.png")

# Fig 2
$tmp = New-Canvas "数据预处理与多源特征工程流程图" (Join-Path $dir "patent_fig2_feature_engineering.png"); $bmp=$tmp[0]; $g=$tmp[1]
Draw-Box $g 590 120 420 70 "日用电量序列" $blue $penBlue $font $brushText
Draw-Box $g 160 260 300 78 "缺失值识别\n零值识别" $orange $penOrange $smallFont $brushText
Draw-Box $g 650 260 300 78 "线性插值\n列均值填充" $orange $penOrange $smallFont $brushText
Draw-Box $g 1140 260 300 78 "完整用电序列" $orange $penOrange $smallFont $brushText
Draw-Arrow $g 800 190 310 260 $arrowPen; Draw-Arrow $g 800 190 800 260 $arrowPen; Draw-Arrow $g 800 190 1290 260 $arrowPen
Draw-Box $g 590 420 420 72 "月度聚合" $green $penGreen $font $brushText
Draw-Arrow $g 1290 338 1010 456 $arrowPen
$features = @("月均值 / 月标准差 / 月峰值", "零值比例 / 缺失比例", "基准偏离 / 累计下降", "跨用户排名 / 排名偏差", "日历季节 / 深层统计", "多尺度变点特征")
$fx = @(130,580,1030,130,580,1030); $fy=@(560,560,560,710,710,710)
for($i=0;$i -lt 6;$i++){ Draw-Box $g $fx[$i] $fy[$i] 390 80 $features[$i] $gray $penBlue $smallFont $brushText; Draw-Arrow $g 800 492 ($fx[$i]+195) $fy[$i] $arrowPen }
Save-Canvas $bmp $g (Join-Path $dir "patent_fig2_feature_engineering.png")

# Fig 3
$tmp = New-Canvas "随机矩阵理论信号增强模块结构图" (Join-Path $dir "patent_fig3_rmt_module.png"); $bmp=$tmp[0]; $g=$tmp[1]
$items=@("用户月度用电矩阵", "鲁棒标准化", "协方差矩阵构建", "特征值分解", "Marchenko-Pastur 谱边界判定", "信号子空间 / 噪声子空间划分")
$y=130
foreach($t in $items){ Draw-Box $g 500 $y 600 70 $t $purple $penPurple $font $brushText; if($y -lt 680){ Draw-Arrow $g 800 ($y+70) 800 ($y+110) $arrowPen }; $y += 110 }
Draw-Box $g 230 810 430 80 "全局信噪比特征" $green $penGreen $font $brushText
Draw-Box $g 940 810 430 80 "后期信号能量占比特征" $green $penGreen $font $brushText
Draw-Arrow $g 800 750 445 810 $arrowPen; Draw-Arrow $g 800 750 1155 810 $arrowPen
Save-Canvas $bmp $g (Join-Path $dir "patent_fig3_rmt_module.png")

# Fig 4
$tmp = New-Canvas "双路径 Transformer 深度时序表征提取网络结构图" (Join-Path $dir "patent_fig4_dual_path_transformer.png"); $bmp=$tmp[0]; $g=$tmp[1]
Draw-Box $g 520 120 560 70 "月度多通道序列输入" $blue $penBlue $font $brushText
Draw-Box $g 520 245 560 70 "输入投影 + 位置编码" $blue $penBlue $font $brushText
Draw-Arrow $g 800 190 800 245 $arrowPen
Draw-Box $g 180 430 500 130 "局部窗口注意力分支\n捕获邻近月份异常波动" $orange $penOrange $font $brushText
Draw-Box $g 920 430 500 130 "全局 Transformer 分支\n捕获全周期长期依赖" $purple $penPurple $font $brushText
Draw-Arrow $g 800 315 430 430 $arrowPen; Draw-Arrow $g 800 315 1170 430 $arrowPen
Draw-Box $g 290 660 360 80 "局部池化特征" $green $penGreen $font $brushText
Draw-Box $g 950 660 360 80 "CLS 全局特征" $green $penGreen $font $brushText
Draw-Arrow $g 430 560 470 660 $arrowPen; Draw-Arrow $g 1170 560 1130 660 $arrowPen
Draw-Box $g 520 830 560 80 "局部—全局拼接深度时序表征" $blue $penBlue $font $brushText
Draw-Arrow $g 470 740 800 830 $arrowPen; Draw-Arrow $g 1130 740 800 830 $arrowPen
Save-Canvas $bmp $g (Join-Path $dir "patent_fig4_dual_path_transformer.png")

# Fig 5
$tmp = New-Canvas "Transformer 表征压缩与 GBDT 集成判别模块结构图" (Join-Path $dir "patent_fig5_fusion_ensemble.png"); $bmp=$tmp[0]; $g=$tmp[1]
Draw-Box $g 150 130 360 80 "基础人工特征" $blue $penBlue $font $brushText
Draw-Box $g 620 130 360 80 "RMT 增强特征" $purple $penPurple $font $brushText
Draw-Box $g 1090 130 360 80 "Transformer 256维表征" $orange $penOrange $font $brushText
Draw-Box $g 1090 280 360 70 "PCA 16维表征" $orange $penOrange $font $brushText
Draw-Arrow $g 1270 210 1270 280 $arrowPen
Draw-Box $g 520 450 560 80 "最终融合特征矩阵" $green $penGreen $font $brushText
Draw-Arrow $g 330 210 650 450 $arrowPen; Draw-Arrow $g 800 210 800 450 $arrowPen; Draw-Arrow $g 1270 350 950 450 $arrowPen
Draw-Box $g 220 640 300 80 "CatBoost" $gray $penBlue $font $brushText
Draw-Box $g 650 640 300 80 "XGBoost" $gray $penBlue $font $brushText
Draw-Box $g 1080 640 300 80 "LightGBM" $gray $penBlue $font $brushText
Draw-Arrow $g 800 530 370 640 $arrowPen; Draw-Arrow $g 800 530 800 640 $arrowPen; Draw-Arrow $g 800 530 1230 640 $arrowPen
Draw-Box $g 520 830 560 80 "AUC² 加权排序融合 → 异常风险分数" $green $penGreen $font $brushText
Draw-Arrow $g 370 720 800 830 $arrowPen; Draw-Arrow $g 800 720 800 830 $arrowPen; Draw-Arrow $g 1230 720 800 830 $arrowPen
Save-Canvas $bmp $g (Join-Path $dir "patent_fig5_fusion_ensemble.png")
