#!/usr/bin/env python3
"""
Script de Verificación del Dataset
Verifica que el dataset esté correctamente estructurado antes del entrenamiento.
"""

import os
import sys
from pathlib import Path
from PIL import Image

def verify_dataset(dataset_root: str):
    """Verifica la estructura y contenido del dataset."""
    print("=" * 60)
    print("🔍 VERIFICACIÓN DEL DATASET")
    print("=" * 60)
    
    # Verificar estructura de carpetas
    required_folders = ['Train', 'Validation', 'Test']
    required_classes = ['Real', 'Fake']
    
    errors = []
    warnings = []
    stats = {}
    
    # Verificar que exista la carpeta raíz
    if not os.path.exists(dataset_root):
        print(f"❌ ERROR: No se encontró la carpeta: {dataset_root}")
        return False
    
    print(f"✅ Carpeta raíz encontrada: {dataset_root}\n")
    
    # Verificar cada split (Train, Validation, Test)
    for split in required_folders:
        split_path = os.path.join(dataset_root, split)
        
        if not os.path.exists(split_path):
            errors.append(f"❌ No existe la carpeta: {split_path}")
            continue
        
        print(f"📁 {split}/")
        stats[split] = {}
        
        # Verificar clases (Real, Fake)
        for class_name in required_classes:
            class_path = os.path.join(split_path, class_name)
            
            if not os.path.exists(class_path):
                errors.append(f"❌ No existe: {split}/{class_name}/")
                continue
            
            # Contar imágenes
            image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
            images = []
            for ext in image_extensions:
                images.extend(Path(class_path).glob(f'*{ext}'))
            
            image_count = len(images)
            stats[split][class_name] = image_count
            
            if image_count == 0:
                warnings.append(f"⚠️  {split}/{class_name}/ está vacía")
                print(f"   {class_name}: 0 imágenes ❌")
            else:
                print(f"   {class_name}: {image_count:,} imágenes ✅")
                
                # Verificar algunas imágenes
                invalid_images = 0
                sample_size = min(10, image_count)
                for img_path in list(images)[:sample_size]:
                    try:
                        img = Image.open(img_path)
                        img.verify()
                        if img.size[0] < 224 or img.size[1] < 224:
                            warnings.append(f"⚠️  {img_path.name} es muy pequeña: {img.size}")
                    except Exception as e:
                        invalid_images += 1
                        warnings.append(f"⚠️  {img_path.name} no se puede abrir: {str(e)}")
                
                if invalid_images > 0:
                    print(f"      ⚠️  {invalid_images} imágenes inválidas encontradas")
        
        print()
    
    # Calcular estadísticas
    print("=" * 60)
    print("📊 ESTADÍSTICAS DEL DATASET")
    print("=" * 60)
    
    total_real = sum(stats.get(split, {}).get('Real', 0) for split in required_folders)
    total_fake = sum(stats.get(split, {}).get('Fake', 0) for split in required_folders)
    total_images = total_real + total_fake
    
    print(f"Total de imágenes: {total_images:,}")
    print(f"  Real: {total_real:,} ({total_real/total_images*100:.1f}%)")
    print(f"  Fake: {total_fake:,} ({total_fake/total_images*100:.1f}%)")
    print()
    
    for split in required_folders:
        if split in stats:
            split_real = stats[split].get('Real', 0)
            split_fake = stats[split].get('Fake', 0)
            split_total = split_real + split_fake
            if split_total > 0:
                print(f"{split}:")
                print(f"  Total: {split_total:,} imágenes")
                print(f"  Real: {split_real:,} ({split_real/split_total*100:.1f}%)")
                print(f"  Fake: {split_fake:,} ({split_fake/split_total*100:.1f}%)")
                
                # Verificar balance
                if abs(split_real - split_fake) / split_total > 0.2:
                    warnings.append(f"⚠️  {split} está desbalanceado (>20% diferencia)")
    
    print()
    
    # Verificar requisitos mínimos
    print("=" * 60)
    print("✅ VERIFICACIÓN DE REQUISITOS")
    print("=" * 60)
    
    min_per_class = 5000
    train_real = stats.get('Train', {}).get('Real', 0)
    train_fake = stats.get('Train', {}).get('Fake', 0)
    
    if train_real >= min_per_class:
        print(f"✅ Train/Real: {train_real:,} (requisito: {min_per_class:,})")
    else:
        errors.append(f"❌ Train/Real: {train_real:,} (requisito mínimo: {min_per_class:,})")
        print(f"❌ Train/Real: {train_real:,} (requisito: {min_per_class:,})")
    
    if train_fake >= min_per_class:
        print(f"✅ Train/Fake: {train_fake:,} (requisito: {min_per_class:,})")
    else:
        errors.append(f"❌ Train/Fake: {train_fake:,} (requisito mínimo: {min_per_class:,})")
        print(f"❌ Train/Fake: {train_fake:,} (requisito: {min_per_class:,})")
    
    # Verificar balance
    if train_real > 0 and train_fake > 0:
        balance_ratio = min(train_real, train_fake) / max(train_real, train_fake)
        if balance_ratio >= 0.8:
            print(f"✅ Balance: {balance_ratio*100:.1f}% (requisito: >80%)")
        else:
            warnings.append(f"⚠️  Balance: {balance_ratio*100:.1f}% (recomendado: >80%)")
            print(f"⚠️  Balance: {balance_ratio*100:.1f}% (recomendado: >80%)")
    
    print()
    
    # Mostrar advertencias
    if warnings:
        print("=" * 60)
        print("⚠️  ADVERTENCIAS")
        print("=" * 60)
        for warning in warnings[:10]:  # Mostrar máximo 10
            print(warning)
        if len(warnings) > 10:
            print(f"... y {len(warnings) - 10} advertencias más")
        print()
    
    # Mostrar errores
    if errors:
        print("=" * 60)
        print("❌ ERRORES")
        print("=" * 60)
        for error in errors:
            print(error)
        print()
        print("❌ El dataset NO está listo para entrenar. Corrige los errores antes de continuar.")
        return False
    else:
        print("=" * 60)
        print("✅ VERIFICACIÓN COMPLETA")
        print("=" * 60)
        print("✅ El dataset está correctamente estructurado y listo para entrenar!")
        print()
        print("💡 Siguiente paso: Ejecuta 'python train_antispoofing.py'")
        return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Verificar estructura del dataset')
    parser.add_argument(
        '--dataset-root',
        type=str,
        default='../Dataset',
        help='Ruta raíz del dataset (default: ../Dataset)'
    )
    
    args = parser.parse_args()
    
    success = verify_dataset(args.dataset_root)
    sys.exit(0 if success else 1)

