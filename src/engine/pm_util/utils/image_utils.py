from PIL import Image


def concatenate_images(image1:Image, image2:Image) -> Image:
    """
    두 개의 PIL 이미지를 가로로 연결하여 새로운 PIL 이미지를 반환합니다.

    Args:
        image1 (PIL.Image.Image): 첫 번째 PIL 이미지.
        image2 (PIL.Image.Image): 두 번째 PIL 이미지.

    Returns:
        PIL.Image.Image: 가로로 연결된 새로운 PIL 이미지.
    """
    # 두 이미지의 높이 중 더 큰 값을 선택합니다.
    max_height = max(image1.height, image2.height)
    
    # 두 이미지의 너비를 합산하여 새로운 이미지의 너비를 계산합니다.
    total_width = image1.width + image2.width
    
    # 두 이미지의 모드가 다르면 오류를 방지하기 위해 공통 모드로 변환합니다.
    # 예: 'RGB' 또는 'RGBA'
    mode = image1.mode
    if image1.mode != image2.mode:
        print("경고: 두 이미지의 모드가 다릅니다. 첫 번째 이미지의 모드에 맞춰 변환합니다.")
        image2 = image2.convert(mode)

    # 새로운 이미지를 생성합니다. 배경은 검은색(0, 0, 0)으로 설정합니다.
    new_image = Image.new(mode, (total_width, max_height), color='black')
    
    # 첫 번째 이미지를 새 이미지의 (0, 0) 위치에 붙여넣습니다.
    new_image.paste(image1, (0, 0))
    
    # 두 번째 이미지를 첫 번째 이미지의 너비만큼 떨어진 위치에 붙여넣습니다.
    new_image.paste(image2, (image1.width, 0))
    
    return new_image