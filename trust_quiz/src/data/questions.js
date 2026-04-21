/**
 * Quiz question bank — unified schema (DB-ready)
 *
 * Every question has the same shape:
 *
 *  {
 *    id:       string           – unique identifier (maps to DB primary key)
 *    question: string           – prompt shown at the top
 *    layout:   'dual'|'single'  – how the screen is arranged
 *    correct:  'left'|'right'   – which option is correct
 *
 *    media: null | { type, src }
 *      – null for 'dual' layout
 *      – the shared media shown in 'single' layout (judge real/fake)
 *
 *    options: {
 *      left:  { label, type, src }   – left panel / left tilt choice
 *      right: { label, type, src }   – right panel / right tilt choice
 *    }
 *      type: 'text' | 'image' | 'video' | 'audio'
 *      src:  URL string for media, null for text
 *  }
 *
 * DB mapping (relational):
 *   questions(id, question, layout, correct, media_type, media_src)
 *   options(question_id, side, label, type, src)
 */

const QUESTIONS = [
  {
    id:       'q_email_1',
    question: 'Email nào là thật',
    layout:   'dual',
    correct:  'left',
    media:    null,
    options: {
      left:  { label: 'Email A', type: 'image', src: '/images/question1/a.png' },
      right: { label: 'Email B', type: 'image', src: '/images/question1/b.png' },
    },
    explanation: {
      heading:   'Email A mới là thật',
      tactic:    'Chiêu "Lợi dụng tiểu tiết"',
      body:      'Dùng các chi tiét cực nhỏ như "rn" đặt sát nhau để trông giống như chữ "m". Khi nhìn lướt (.corn thay vì .com), não sẽ tự đọc thành chữ quen.',
      tipsLabel: 'Cách nhận biết email giả:',
      tips: [
        'Đọc kỹ từng ký tự trong đường link trước khi nhấp, không tin vào cái nhìn lướt',
      ],
    },
  },
  {
    id:       'q_fanpage_1',
    question: 'Đâu là Fanpage thật?',
    layout:   'dual',
    correct:  'left',
    media:    null,
    options: {
      left:  { label: 'Fanpage A', type: 'image', src: '/images/question0/a.png' },
      right: { label: 'Fanpage B', type: 'image', src: '/images/question0/b.png' },
    },
    explanation: {
      heading:   'Fanpage B là bẫy giả mạo',
      tactic:    'Chiêu "Ve Sầu Thoát Xác" (Đổi tên trang)',
      body:      'Mua fanpage cũ, đổi tên thành thương hiệu uy tín để xoá dấu vết. Đẩy lượt thích trang cá nhân vượt trang chính, giả tick xanh, dựng cảm giác đã được nhiều người tin.',
      tipsLabel: 'Nạn nhân có thể phát hiện thật-giả bằng cách:',
      tips: [
        'Kiểm tra mục Lịch sử Trang',
        'Phát hiện trang vừa đổi tên gần đây',
        'Nhận ra dấu hiệu bất thường từ lịch sử đổi tên: tên cũ lộn xộn, nhiều hạng mục không liên quan',
      ],
    },
  },
  {
    id:       'q_fanpage_2',
    question: 'Website nào được dựng lên để đánh lừa người dùng?',
    layout:   'dual',
    correct:  'left',
    media:    null,
    options: {
      left:  { label: 'Website A', type: 'image', src: '/images/question2/a.png' },
      right: { label: 'Website B', type: 'image', src: '/images/question2/b.png' },
    },
    explanation: {
      heading:   'Website A là giả',
      tactic:    'Chiêu thao túng bằng vẻ ngoài',
      body:      'Dựng website giả mạo thật Nạn nhân có thể phát hiện bằng cách:',
      tips: [
        'Bỏ qua phần nhìn, kiểm tra kỹ tên miền (URL), đồng thời xác minh thêm: gọi số liên hệ, hỏi các group uy tín'
      ],
    },
  },
  {
    id:       'q_warrant_1',
    question: 'Đâu là lệnh bắt tạm giam giả?',
    layout:   'dual',
    correct:  'left',
    media:    null,
    options: {
      left:  { label: 'Image A', type: 'image', src: '/images/question4/a.png' },
      right: { label: 'Image B', type: 'image', src: '/images/question4/b.png' },
    },
    explanation: {
      heading:   'Lệnh B là lệnh giả để tạo áp lực',
      tactic:    '"Giả Công An" (Authority Impersonation)',
      body:      'Chỉ cần dựng một văn bản nhìn "chuẩn không cần chỉnh", bố cục gọn gàng. \nNội dung thì không cần dài dòng, vài điều khoản ngắn gọn nhưng đủ lực, đọc vào là thấy áp lực tự tới',
      tipsLabel: 'Dấu hiệu nhận biết giấy tờ giả:',
      tips: [
        'Chủ động gọi cơ quan liên quan để xác minh',
        'Không hoảng sợ mà tin mình vi phạm',
        'Giữ bình tĩnh để kiểm tra kỹ thông tin, quốc hiệu, tên cơ quan, số văn bản, ngày tháng, tiêu đề, căn cứ pháp lý và các điều khoản',
      ],
    },
  },
  {
    id:       'q_single_1',
    question: 'Thông tin trong ảnh này là thật hay giả?',
    layout:   'single',
    correct:  'left',
    media:    { type: 'image', src: '/images/question5/a.png' },
    options: {
      left:  { label: 'Thật', type: 'text', src: null },
      right: { label: 'Giả',  type: 'text', src: null },
    },
    explanation: {
      heading:   'Hình trên là email thật với tên miền đúng của Shopee',
      tactic:    'Chiêu gài tên miền lạ',
      body:      'Chỉ cần lấy link quen thuộc rồi biến tấu nhẹ tên miền - thêm đuôi (.vip, .top, .xyz,...) nhìn vẫn na ná link thật. Dùng để lấy thông tin đăng nhập, xin OTP, hoặc khiến họ tự chuyển tiền.',
      tipsLabel: 'Nạn nhân có thể phát hiện thật-giả bằng cách:',
      tips: [
        'Đối tượng bị nhắm tới nhìn kỹ toàn bộ tên miền, không chỉ phần đầu. Thấy duôi lạ hoặc cấu trúc bất thường là dừng'
      ],
    },
  },
  {
    id:       'q_video_2',
    question: 'Đâu là người thật?',
    layout:   'dual',
    correct:  'left',
    media:    null,
    options: {
      left:  { label: 'Video A', type: 'video', src: '/videos/question2/a.mp4' },
      right: { label: 'Video B', type: 'video', src: '/videos/question2/b.mp4' },
    },
    explanation: {
      heading:   'Video A là clip nhân vật thật',
      tactic:    'Chiêu "khoác mặt người nổi tiếng"',
      body:      'Dùng AI deepfake giả cả hình lẫn giọng để lừa tuyển dụng, bán hàng,...\nTinh vi đến mức khó phân biệt, nên chiêu này dùng đâu cũng hiệu quả.',
      tipsLabel: 'Nạn nhân có thể phát hiện thật-giả bằng cách:',
      tips: [
        'Yêu cầu xác minh thêm bằng video, kiểm tra chéo trên nhiều nền tảng và kênh chính thức',
      ],
    },
  },
  {
    id:       'q_video_1',
    question: 'Đâu là IShowSpeed thật?',
    layout:   'dual',
    correct:  'right',
    media:    null,
    options: {
      left:  { label: 'Video A', type: 'video', src: '/videos/question1/a.mp4' },
      right: { label: 'Video B', type: 'video', src: '/videos/question1/b.mp4' },
    },
    explanation: {
      heading:   'Video B là mới là IShowSpeed thật',
      tactic:    '"Mạo Danh Người Nổi Tiếng" (Celebrity Impersonation)',
      body:      'Kẻ gian dùng AI tạo video giả của người nổi tiếng để quảng cáo đầu tư, cờ bạc online, hoặc sản phẩm kém chất lượng. Nạn nhân tin tưởng vì thấy thần tượng "xác nhận".',
      tipsLabel: 'Cách kiểm tra video người nổi tiếng:',
      tips: [
        'Tìm video gốc trên kênh chính thức được xác minh (tick xanh)',
        'Nếu có kêu gọi đầu tư hoặc nhận quà → chắc chắn là lừa đảo',
        'Báo cáo video giả cho nền tảng để bảo vệ người khác',
      ],
    },
  },
  {
    id:       'q_voice_1',
    question: 'Đâu là giọng giả mà vẫn quen tai?',
    layout:   'dual',
    correct:  'right',
    media:    null,
    options: {
      left:  { label: 'Voice A', type: 'audio', src: '/audios/audio_a.mp3' },
      right: { label: 'Voice B', type: 'audio', src: '/audios/audio_a.mp3' },
    },
    explanation: {
      heading:   'Voice B là giọng được tạo bởi AI',
      tactic:    'Chiêu vay mượn giọng nói',
      body:      'Tái tạo giọng của một người bằng AI, rồi dùng chính giọng đó để gọi điện lừa đảo.',
      tipsLabel: 'Nạn nhân có thể phát hiện thật-giả bằng cách:',
      tips: [
        'Đề phòng cảm giác quen tai: đừng chỉ tin giọng quen',
        'Yêu cầu xác minh thêm bằng video',
        'Tuyệt đối không chuyển tiền chỉ vì nghe thấy giọng quen',
      ],
    },
  },
  {
    id:       'q_video_3',
    question: 'Đâu là video "câu nước mắt" để lừa đảo kêu gọi quyên góp?',
    layout:   'dual',
    correct:  'right',
    media:    null,
    options: {
      left:  { label: 'Video A', type: 'video', src: '/videos/question3/a.mp4' },
      right: { label: 'Video B', type: 'video', src: '/videos/question3/b.mp4' },
    },
    explanation: {
      heading:   'Video B là video từ thiện giả',
      tactic:    '"Câu Nước Mắt" (Emotional Manipulation)',
      body:      'Kẻ gian dùng hình ảnh trẻ em, người già bệnh tật — thường lấy từ nguồn khác hoặc dàn dựng — để kêu gọi quyên góp. Tiền không đến tay người cần mà vào túi kẻ lừa đảo.',
      tipsLabel: 'Trước khi quyên góp, hãy:',
      tips: [
        'Tìm tên tổ chức từ thiện trên Google — có đăng ký hợp pháp không?',
        'Dùng Google Reverse Image Search để kiểm tra ảnh có bị lấy từ nơi khác không',
        'Ưu tiên quyên góp qua các quỹ từ thiện được Nhà nước công nhận',
      ],
    },
  },
  {
    id:       'q_video_4',
    question: 'Clip nào có thể là bẫy giúp đỡ?',
    layout:   'dual',
    correct:  'left',
    media:    null,
    options: {
      left:  { label: 'Video A', type: 'video', src: '/videos/question4/a.mp4' },
      right: { label: 'Video B', type: 'video', src: '/videos/question4/b.mp4' },
    },
    explanation: {
      heading:   'Video A là bẫy "giả vờ cần giúp đỡ"',
      tactic:    '"Bẫy Lòng Tốt" (Good Samaritan Scam)',
      body:      'Kẻ gian giả vờ gặp khó khăn — hết xăng, lạc đường, mất ví — để tiếp cận và xin tiền. Sau khi nhận tiền, chúng biến mất hoặc quay lại với bẫy tinh vi hơn.',
      tipsLabel: 'Khi gặp người lạ xin giúp đỡ:',
      tips: [
        'Không đưa tiền mặt — đề nghị hỗ trợ theo cách khác (gọi xe, liên hệ gia đình)',
        'Cảnh giác nếu họ từ chối mọi hình thức hỗ trợ trừ tiền mặt',
        'Tin vào trực giác — nếu cảm thấy bất an, hãy rời đi an toàn',
      ],
    },
  },


  // ── Dual layout: text only ───────────────────────────────────────
  // {
  //   id:       'q_text_1',
  //   question: 'AI stands for?',
  //   layout:   'dual',
  //   correct:  'left',
  //   media:    null,
  //   options: {
  //     left:  { label: 'Artificial Intelligence', type: 'text', src: null },
  //     right: { label: 'Automated Input',          type: 'text', src: null },
  //   },
  // },
];

export default QUESTIONS;
